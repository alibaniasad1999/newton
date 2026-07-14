# PhD Cable Project - Newton simulation scripts
#
# Two Franka Panda (FR3) arms face each other and hold the two ends of a
# charger-cable-like cable, one end pinched in each gripper. Unlike the
# single-arm scripts in this project (which freeze the arm links and drive
# them kinematically with newton.eval_fk), BOTH arms here are fully dynamic
# articulations solved by SolverVBD: every link keeps its real URDF
# mass/inertia, every joint is a real revolute/prismatic joint with PD
# position drives (model.joint_target_ke/kd) and URDF limits.
#
# Arm A (the "mover") gets stiff drives and its shoulder-joint PD *target*
# is swept back and forth with a sine wave -- a realistic position-servo
# tracking a commanded trajectory, so the arm lags/overshoots slightly like
# a real robot instead of teleporting. Arm B (the "holder") gets much
# softer drives whose targets stay at its initial pose: when arm A pulls
# away, cable tension genuinely drags arm B's joints off their targets, and
# when arm A comes back, arm B's PD servos pull it back toward its pose.
# Force transmission is two-way and entirely physical: hand -> ball joint
# -> cable stretch -> ball joint -> other hand -> joint torques.
#
# Cable attachment: each cable end is connected to its hand by a ball
# joint at the gripper TCP (the point between the fingertips), modelling a
# firm pinch that still lets the cable pivot. The two ball joints close a
# kinematic loop (world -> arm A -> cable -> arm B -> world), so they are
# deliberately NOT added to any articulation: newton.eval_fk skips
# loop-closure joints, while SolverVBD still solves them as hard
# constraints.
#
# The URDFs are imported with collapse_fixed_joints=True so the massless
# fixed-joint links (fr3_link8, fr3_hand_tcp) are merged into their parents
# -- a zero-mass body is immovable to the solver and would otherwise anchor
# the dynamic arm in place through its fixed joint.
#
# Arm mesh collision is disabled (collision_group=0), as in the other
# scripts: full mesh-vs-mesh narrow phase across ~65 Franka collision
# shapes per arm dominated CPU cost in earlier testing. The grip is
# modelled by the ball joints, so no gripper-cable contact is needed; the
# cable still collides with the ground plane.
#
# Mouse controls (GL viewer): see phd_cable_sim/cube_anchor_cable_swing.py
#
# Run with the GUI:
#   uv run -m phd_cable_sim.franka_two_arm_cable_pull --viewer gl
#
# Run headless with the built-in sanity checks (300 frames = 1 s settle
# + one full 0.25 Hz sweep period):
#   uv run -m phd_cable_sim.franka_two_arm_cable_pull --viewer null --test --num-frames 300

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

# Fixed "reach forward" joint pose (7 arm joints + 2 finger joints). Chosen
# so the gripper's approach axis (the fr3_hand -> TCP direction, i.e.
# "between the fingers") points purely along local +X, not straight down as
# with a naive elbow-bend pose -- otherwise the fingers point at the ground
# and a cable can't thread between them. The finger joints (each a
# 0-0.04 m prismatic slide from the hand's centerline) are nearly closed so
# the gripper visually pinches the 3 mm-radius cable rather than floating
# open around it.
FRANKA_POSE_Q = [0.0, 0.0, 0.0, -1.5708, 0.0, 3.1416, 0.785, 0.004, 0.004]

# PD drive gains, in VBD's convention (kd is absolute damping in physical
# units). The mover's servos are stiff so it tracks its commanded sweep
# against gravity and cable tension; the holder's are ~10x softer so cable
# tension visibly drags it off its targets (it sags a little under gravity
# too, like a compliant real controller without gravity compensation).
MOVER_ARM_KE, MOVER_ARM_KD = 5000.0, 200.0
HOLDER_ARM_KE, HOLDER_ARM_KD = 150.0, 15.0
FINGER_KE, FINGER_KD = 1000.0, 10.0


@wp.kernel
def sweep_target_kernel(
    target_index: int,
    base_angle: float,
    amplitude: float,
    angular_freq: float,
    start_time: float,
    t: wp.array[float],
    joint_target_q: wp.array[float],
):
    """Sweep one joint's PD position target sinusoidally around base_angle,
    holding still until start_time so the scene first settles under
    gravity. Only the commanded target moves; the joint itself follows via
    its drive, lagging realistically under load."""
    phase = wp.max(t[0] - start_time, 0.0)
    joint_target_q[target_index] = base_angle + amplitude * wp.sin(angular_freq * phase)


@wp.kernel
def advance_time_kernel(t: wp.array[float], dt: float):
    t[0] = t[0] + dt


def _find_label(labels, suffix: str, start: int, end: int) -> int:
    for i in range(start, end):
        if labels[i].endswith(suffix):
            return i
    raise RuntimeError(f"Could not find a label ending with '{suffix}' among [{start}, {end})")


def _resolve_tcp_local_poses(urdf_path):
    """Resolve, for FRANKA_POSE_Q on an unpositioned single-arm model, the
    fr3_hand_tcp transform (a) in the arm-base frame and (b) in the
    fr3_link7 frame, via a throwaway finalize + forward-kinematics pass.

    The probe imports WITHOUT collapse_fixed_joints so the fr3_hand_tcp
    body still exists to be located; the link7-relative transform is what
    remains valid on the collapsed model used in the real scene, where the
    hand/TCP links are merged into fr3_link7 (whose body frame collapse
    preserves).
    """
    probe_builder = newton.ModelBuilder()
    probe_builder.add_urdf(urdf_path, floating=False)
    probe_model = probe_builder.finalize()
    probe_state = probe_model.state()
    q = wp.array(FRANKA_POSE_Q, dtype=wp.float32)
    newton.eval_fk(probe_model, q, probe_model.joint_qd, probe_state)
    body_q = probe_state.body_q.numpy()

    def world_xform(suffix):
        pose = body_q[_find_label(probe_builder.body_label, suffix, 0, probe_builder.body_count)]
        return wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))

    tcp_in_base = world_xform("fr3_hand_tcp")
    link7_in_base = world_xform("fr3_link7")
    tcp_in_link7 = wp.transform_multiply(wp.transform_inverse(link7_in_base), tcp_in_base)
    return tcp_in_base, tcp_in_link7


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        # Simulation cadence. More VBD iterations than the single-arm
        # scripts (10 vs 5): the hard ball joints couple a ~1.3 kg hand to
        # ~0.2 g cable segments and need the extra convergence.
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Charger-cable-like properties: thin, light, floppy. add_rod takes
        # PER-JOINT spring constants, so material stiffness converts as
        # k = EA/L_seg (stretch) and EI/L_seg (bend); with 24 segments over
        # ~0.9 m (L_seg ~ 3.75 cm), the values below correspond to
        # EA ~ 3.8e4 N and EI ~ 3.8e-3 N*m^2 -- a realistic copper-cored
        # 3 mm charger cable (nearly inextensible, floppy in bending). The
        # stiff stretch also transmits the arm-A pull to arm B almost
        # losslessly instead of soaking it up as stretch.
        self.num_elements = 24
        cable_radius = 0.003  # 3 mm
        stretch_stiffness = 1.0e6  # [N/m] per joint
        stretch_damping = 1.0e-4  # [N*s/m]
        bend_stiffness = 0.1  # [N*m/rad] per joint
        bend_damping = 5.0e-2  # [N*m*s/rad]

        builder = newton.ModelBuilder()

        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0

        cable_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=200.0,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
        )

        urdf_path = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"

        tcp_in_base, tcp_in_link7 = _resolve_tcp_local_poses(urdf_path)

        # The two arms face each other along X, bases placed so the TCPs
        # end up ~0.9 m apart with the cable strung taut between them.
        tcp_span = 0.9
        tcp_local_p = wp.transform_get_translation(tcp_in_base)
        base_half_dist = 0.5 * tcp_span + tcp_local_p[0]
        xform_a = wp.transform(
            wp.vec3(base_half_dist, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        )
        xform_b = wp.transform(wp.vec3(-base_half_dist, 0.0, 0.0), wp.quat_identity())

        def add_arm(base_xform, label_prefix, arm_ke, arm_kd):
            """Add one fully dynamic Franka: real URDF masses, PD drives on
            all 9 dofs targeting FRANKA_POSE_Q, URDF joint limits kept."""
            body_start = builder.body_count
            shape_start = builder.shape_count
            joint_start = builder.joint_count
            sub = newton.ModelBuilder()
            # collapse_fixed_joints merges the massless fr3_link8 /
            # fr3_hand_tcp links (immovable to the solver) into fr3_link7.
            sub.add_urdf(
                urdf_path,
                floating=False,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
            )
            sub.joint_q = list(FRANKA_POSE_Q)
            sub.joint_target_q = list(FRANKA_POSE_Q)
            sub.joint_target_ke = [arm_ke] * 7 + [FINGER_KE] * 2
            sub.joint_target_kd = [arm_kd] * 7 + [FINGER_KD] * 2
            builder.add_builder(sub, xform=base_xform, label_prefix=label_prefix)

            # Disable arm mesh collision (see header); the grip is modelled
            # by the TCP ball joint, not by finger contact.
            for shape in range(shape_start, builder.shape_count):
                builder.shape_collision_group[shape] = 0

            hand = _find_label(builder.body_label, "fr3_link7", body_start, builder.body_count)
            return hand, joint_start

        hand_a, joint_start_a = add_arm(xform_a, "arm_a", MOVER_ARM_KE, MOVER_ARM_KD)
        hand_b, _joint_start_b = add_arm(xform_b, "arm_b", HOLDER_ARM_KE, HOLDER_ARM_KD)
        self.hand_a = hand_a
        self.hand_b = hand_b

        # Arm A's swept shoulder: fr3_joint2. The sweep is applied to the
        # PD *target*, so the arm is force-driven, never teleported.
        self.shoulder_joint_a = _find_label(builder.joint_label, "fr3_joint2", joint_start_a, builder.joint_count)
        self.shoulder_base_angle = FRANKA_POSE_Q[1]
        # Negative amplitude so the first half-period pulls AWAY from arm B
        # (taut cable drags it), the second gives slack (cable droops).
        # Well within fr3_joint2's +/-1.784 rad limit; how far arm B gets
        # dragged is set by the MOVER/HOLDER gain ratio above, not by any
        # joint limit.
        self.sweep_amplitude = -0.8  # [rad]
        self.sweep_angular_freq = 2.0 * np.pi * 0.25  # 0.25 Hz
        # Hold still for the first second so the softly-servoed holder arm
        # settles under gravity before the pulling starts.
        self.sweep_start_time = 1.0  # [s]

        # TCP world positions for the initial pose, to string the cable.
        tcp_a = wp.transform_multiply(xform_a, tcp_in_base)
        tcp_b = wp.transform_multiply(xform_b, tcp_in_base)
        tcp_a_p = wp.transform_get_translation(tcp_a)
        tcp_b_p = wp.transform_get_translation(tcp_b)
        cable_vec = tcp_a_p - tcp_b_p
        cable_length = float(wp.length(cable_vec))
        segment_length = cable_length / self.num_elements

        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=tcp_b_p,
            direction=cable_vec,
            length=cable_length,
            num_segments=self.num_elements,
        )

        cable_bodies, _cable_joints = builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=cable_radius,
            cfg=cable_shape_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            label="charger_cable",
            body_frame_origin="com",
        )
        self.cable_bodies = cable_bodies

        # Pinch each cable end to its gripper TCP with a ball joint. These
        # close the world->armA->cable->armB->world kinematic loop, so they
        # must stay OUT of any articulation (eval_fk skips them; SolverVBD
        # still enforces them). body_frame_origin="com" puts each segment's
        # body origin at its midpoint, so the end attach points sit half a
        # segment from the body origin along the rod's local Z.
        builder.add_joint_ball(
            parent=hand_b,
            child=cable_bodies[0],
            parent_xform=tcp_in_link7,
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * segment_length), wp.quat_identity()),
        )
        builder.add_joint_ball(
            parent=hand_a,
            child=cable_bodies[-1],
            parent_xform=tcp_in_link7,
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * segment_length), wp.quat_identity()),
        )

        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize()

        # SolverVBD uses model.body_q as the structural rest pose, so bring
        # it in line with the FRANKA_POSE_Q joint angles before
        # constructing the solver (add_urdf does not run FK).
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.sim_iterations, rigid_body_contact_buffer_size=256
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Slot of arm A's shoulder dof inside control.joint_target_q,
        # resolved through the model's per-joint target layout so it works
        # for both dof- and coord-shaped target arrays.
        self.shoulder_target_index = int(self.model.joint_target_q_start.numpy()[self.shoulder_joint_a])

        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.viewer.set_model(self.model)

        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            ps = picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 20.0
            ps[0]["pick_damping"] = 1.0
            picking.pick_state.assign(ps)

        # Face-on view of the x-z plane the arms and cable move in (Z-up yaw
        # convention: front = (cos yaw, sin yaw)); pulled back along -Y far
        # enough to frame both arms.
        center = 0.5 * (tcp_a_p + tcp_b_p)
        self.viewer.set_camera(pos=wp.vec3(center[0], center[1] - 2.5, center[2]), pitch=0.0, yaw=90.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.set_pivot(center)

        # Test bookkeeping: hand B's gravity-settled pose is captured as a
        # baseline once the sweep starts; the largest excursion from it,
        # recorded by test_post_step, is proof the cable really pulled the
        # holder arm (not just that it sagged under gravity).
        self.hand_b_baseline = None
        self.max_hand_b_excursion = 0.0

        self.capture()

    def capture(self):
        """Capture simulation loop into a graph for optimal performance."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for _substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                sweep_target_kernel,
                dim=1,
                inputs=[
                    self.shoulder_target_index,
                    self.shoulder_base_angle,
                    self.sweep_amplitude,
                    self.sweep_angular_freq,
                    self.sweep_start_time,
                    self.sim_time_array,
                ],
                outputs=[self.control.joint_target_q],
                device=self.solver.device,
            )
            wp.launch(
                advance_time_kernel,
                dim=1,
                inputs=[self.sim_time_array, self.sim_dt],
                device=self.solver.device,
            )

            self.model.collide(self.state_0, self.contacts)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Track how far the holder arm's hand gets dragged from its
        gravity-settled baseline once the sweep is underway."""
        if self.sim_time < self.sweep_start_time:
            return
        hand_b_pos = np.array(self.state_0.body_q.numpy()[self.hand_b][:3])
        if self.hand_b_baseline is None:
            self.hand_b_baseline = hand_b_pos
            return
        excursion = float(np.linalg.norm(hand_b_pos - self.hand_b_baseline))
        self.max_hand_b_excursion = max(self.max_hand_b_excursion, excursion)

    def test_final(self):
        """Verify stability, attachment, and that the cable pulled arm B."""
        body_positions = self.state_0.body_q.numpy()
        body_velocities = self.state_0.body_qd.numpy()

        assert np.isfinite(body_positions).all(), "Non-finite positions"
        assert np.isfinite(body_velocities).all(), "Non-finite velocities"
        assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"

        # Both cable ends should still be pinched at their hands (the ball
        # joints are hard constraints; allow a little solver slack).
        for hand, end_body in ((self.hand_b, self.cable_bodies[0]), (self.hand_a, self.cable_bodies[-1])):
            hand_pos = body_positions[hand][:3]
            end_pos = body_positions[end_body][:3]
            dist = np.linalg.norm(end_pos - hand_pos)
            assert dist < 0.3, f"Cable end detached from hand: {end_pos} vs {hand_pos} (dist {dist:.3f})"

        # The holder arm must actually have been dragged by the cable: its
        # hand should have moved measurably from its gravity-settled pose
        # at some point after the sweep started (its joints are dynamic,
        # only held by soft PD servos).
        assert self.hand_b_baseline is not None, (
            "Run too short for the pull test: needs --num-frames > "
            f"{int(self.sweep_start_time * self.fps)} so the sweep actually starts"
        )
        assert self.max_hand_b_excursion > 0.02, (
            f"Holder arm was never pulled by the cable (max excursion {self.max_hand_b_excursion * 100:.1f} cm)"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
