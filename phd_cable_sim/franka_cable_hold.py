# PhD Cable Project - Newton simulation scripts
#
# One Franka Panda (FR3) arm holds one end of a charger-cable-like cable in
# its hand; the other end is tied to a free-hanging weight (a small box
# with real mass, no attachment to anything else). The arm's shoulder joint
# (fr3_joint2) is swept back and forth with a sine wave each substep,
# driven purely kinematically via newton.eval_fk -- the arm's own joints
# actually articulate, so the hand traces an arc as a consequence of the
# shoulder rotating. The weight is attached to the cable via a real ball
# joint (not kinematically pinned), so cable tension genuinely swings it
# around as the arm moves -- this is the part that actually transmits
# force like a rigid link would, unlike an earlier version where both
# cable ends were kinematically forced and nothing ever really moved in
# response to the other end.
#
# Why not two Frankas, both cable ends actually dynamic? A version with a
# second Franka holding the "stationary" end was tried with that end's
# cable segment as a free (non-pinned) physics body, softly held near the
# hand -- it was never made fully robust (drifted away over a long run,
# never recovered). Swapping that whole arm for a single free weight body
# sidesteps the problem entirely: there is no anchor to fall away from, the
# weight IS the free end.
#
# Mouse controls (GL viewer): see phd_cable_sim/cube_anchor_cable_swing.py
#
# Run with the GUI:
#   uv run -m phd_cable_sim.franka_cable_hold --viewer gl
#
# Run headless with the built-in sanity checks:
#   uv run -m phd_cable_sim.franka_cable_hold --viewer null --test --num-frames 120

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


@wp.kernel
def sweep_joint_kernel(
    dof_index: int,
    base_angle: float,
    amplitude: float,
    angular_freq: float,
    t: wp.array[float],
    joint_q: wp.array[float],
):
    """Sweep a single joint DOF back and forth sinusoidally around
    base_angle, writing the result into the live joint_q buffer that's fed
    to newton.eval_fk every substep."""
    cur_t = t[0]
    joint_q[dof_index] = base_angle + amplitude * wp.sin(angular_freq * cur_t)


@wp.kernel
def track_hand_kernel(
    hand: int,
    cable_end: int,
    end_local: wp.transform,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Keep the pinned cable-end body welded to the hand's current pose.
    end_local is the fixed (rigid-weld) transform of the cable end body
    relative to the hand, computed once at build time."""
    hand_xform = body_q0[hand]
    end_xform = wp.transform_multiply(hand_xform, end_local)
    body_q0[cable_end] = end_xform
    body_q1[cable_end] = end_xform


@wp.kernel
def advance_time_kernel(t: wp.array[float], dt: float):
    t[0] = t[0] + dt


def _find_body(body_label, label_suffix: str, start: int, end: int) -> int:
    for i in range(start, end):
        if body_label[i].endswith(label_suffix):
            return i
    raise RuntimeError(f"Could not find a body ending with '{label_suffix}' among bodies [{start}, {end})")


def _resolve_hand_local_pose(urdf_path):
    """Resolve the fr3_hand_tcp world transform for FRANKA_POSE_Q on an
    unpositioned (identity xform) single-arm model, via a throwaway
    finalize + forward-kinematics pass.

    `ModelBuilder.add_urdf` does not run forward kinematics, so
    `builder.body_q` stays at each body's raw (pre-FK) pose regardless of
    `joint_q` -- only a finalized model's `state.body_q`, after
    `newton.eval_fk`, reflects the joint configuration.
    """
    probe_builder = newton.ModelBuilder()
    probe_builder.add_urdf(urdf_path, floating=False)
    probe_model = probe_builder.finalize()
    probe_state = probe_model.state()
    q = wp.array(FRANKA_POSE_Q, dtype=wp.float32)
    newton.eval_fk(probe_model, q, probe_model.joint_qd, probe_state)
    tcp_index = _find_body(probe_builder.body_label, "fr3_hand_tcp", 0, probe_builder.body_count)
    pose = probe_state.body_q.numpy()[tcp_index]
    return wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Charger-cable-like properties: thin, light, floppy
        self.num_elements = 24
        cable_radius = 0.003  # 3 mm
        stretch_stiffness = 1.0e5
        stretch_damping = 1.0e-4
        bend_stiffness = 1.0e-2
        bend_damping = 1.0e-2

        # Free-hanging weight tied to the cable's far end.
        weight_mass = 0.1  # [kg]
        weight_half_extent = 0.04  # [m]

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

        hand_local = _resolve_hand_local_pose(urdf_path)

        # Arm positioned so the hand sits opposite the weight, ~1 m apart.
        cable_span = 1.0
        arm_base_xform = wp.transform(
            wp.vec3(0.5 * cable_span, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        )
        hand_xform = wp.transform_multiply(arm_base_xform, hand_local)
        hand_p = wp.transform_get_translation(hand_xform)
        weight_p = wp.vec3(-0.5 * cable_span, 0.0, hand_p[2])

        arm_body_start = builder.body_count
        arm_shape_start = builder.shape_count
        arm_dof_start = builder.joint_dof_count
        arm_articulation_index = builder.articulation_count
        arm_sub = newton.ModelBuilder()
        arm_sub.add_urdf(urdf_path, floating=False)
        arm_sub.joint_q = list(FRANKA_POSE_Q)
        builder.add_builder(arm_sub, xform=arm_base_xform, label_prefix="arm")
        arm_body_end = builder.body_count
        arm_shape_end = builder.shape_count
        hand = _find_body(builder.body_label, "arm/fr3/fr3_hand_tcp", arm_body_start, arm_body_end)

        # Hold every arm link fixed by default -- zero mass/inertia keeps
        # the solver from otherwise trying to move them under
        # gravity/contacts. The shoulder joint (dof 1, fr3_joint2) is
        # instead driven kinematically each substep via newton.eval_fk (see
        # simulate()), which is what actually moves the arm's links; the
        # zeroed mass has no effect on bodies whose pose is written
        # directly by eval_fk.
        for body in range(arm_body_start, arm_body_end):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        # Shoulder-joint sweep: fr3_joint2 (dof index 1 within the arm's 9
        # dof) rocks back and forth around its FRANKA_POSE_Q angle, swinging
        # the whole downstream chain (forearm, wrist, hand) through an arc
        # -- real joint motion, not a rigid translation of the arm.
        self.arm_articulation_index = arm_articulation_index
        self.shoulder_dof = arm_dof_start + 1
        self.shoulder_base_angle = FRANKA_POSE_Q[1]
        self.sweep_amplitude = 0.6  # [rad]
        self.sweep_angular_freq = 2.0 * np.pi * 0.25  # 0.25 Hz

        # Arm mesh collision is disabled: full mesh-vs-mesh narrow phase
        # across ~65 Franka collision shapes cost ~34 ms/substep on CPU
        # (vs ~2 ms for the VBD solver step itself) in earlier testing.
        # collision_group=0 means "collides with nothing"; the cable and
        # weight still collide normally with the ground plane.
        for shape in range(arm_shape_start, arm_shape_end):
            builder.shape_collision_group[shape] = 0

        # Free-hanging weight: a real dynamic body (not kinematic), so
        # cable tension genuinely swings it -- this is the part that
        # actually transmits force like a rigid link would.
        weight = builder.add_body(xform=wp.transform(weight_p, wp.quat_identity()), mass=weight_mass)
        builder.add_shape_box(weight, hx=weight_half_extent, hy=weight_half_extent, hz=weight_half_extent)
        self.weight = weight

        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=weight_p,
            direction=hand_p - weight_p,
            length=float(wp.length(hand_p - weight_p)),
            num_segments=self.num_elements,
        )

        cable_bodies, cable_joints = builder.add_rod(
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
            wrap_in_articulation=False,
        )
        self.cable_bodies = cable_bodies
        self.hand = hand

        # Fixed (rigid-weld) transform of the hand-side cable end relative
        # to the hand, computed once here from their initial world poses.
        # Used every substep (see track_hand_kernel) to keep that end
        # welded to the hand as it swings.
        hand_cable_end_xform = wp.transform(*builder.body_q[cable_bodies[-1]])
        self.hand_end_local = wp.transform_multiply(wp.transform_inverse(hand_xform), hand_cable_end_xform)

        # Hand-side end: pinned (zero mass/inertia), driven every substep to
        # track the (swinging) hand -- this is the scripted "active" end.
        hand_end_body = cable_bodies[-1]
        builder.body_mass[hand_end_body] = 0.0
        builder.body_inv_mass[hand_end_body] = 0.0
        builder.body_inertia[hand_end_body] = wp.mat33(0.0)
        builder.body_inv_inertia[hand_end_body] = wp.mat33(0.0)

        # Weight-side end: attached to the free weight body with a ball
        # joint (verified stable in isolated testing -- unlike a
        # zero-mass/near-massless cable segment, a joint to a body with
        # real mass, like this 100 g weight, doesn't destabilize
        # SolverVBD). body_frame_origin="com" places the rod segment's body
        # origin at its midpoint, so the child attach point is half a
        # segment back from the segment's actual start.
        segment_length = float(wp.length(hand_p - weight_p)) / self.num_elements
        weight_end_body = cable_bodies[0]
        j_ball = builder.add_joint_ball(
            parent=weight,
            child=weight_end_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * segment_length), wp.quat_identity()),
        )
        builder.add_articulation([*cable_joints, j_ball])

        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=self.sim_iterations, rigid_body_contact_buffer_size=256
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        # A live, mutable copy of joint_q that simulate() writes the swept
        # shoulder angle into each substep, fed to newton.eval_fk to move
        # the arm. model.joint_q itself is left untouched.
        self.live_joint_q = wp.clone(self.model.joint_q)
        self.arm_articulation_indices = wp.array(
            [self.arm_articulation_index], dtype=wp.int32, device=self.solver.device
        )

        newton.eval_fk(self.model, self.live_joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.live_joint_q, self.model.joint_qd, self.state_1)

        self.viewer.set_model(self.model)

        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            ps = picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 20.0
            ps[0]["pick_damping"] = 1.0
            picking.pick_state.assign(ps)

        # Face-on view of the x-z plane the arm and weight move in (Z-up yaw
        # convention: front = (cos yaw, sin yaw)).
        center = 0.5 * (hand_p + weight_p)
        self.viewer.set_camera(pos=wp.vec3(center[0], center[1] - 2.2, center[2]), pitch=0.0, yaw=90.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.set_pivot(center)

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
            self.model.collide(self.state_0, self.contacts)

            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

            # Sweep the arm's shoulder joint, then re-resolve just that
            # arm's forward kinematics (indices=[...] leaves the cable's
            # and weight's body_q untouched), and re-weld the hand-side
            # cable end to the (now-moved) hand.
            wp.launch(
                sweep_joint_kernel,
                dim=1,
                inputs=[
                    self.shoulder_dof,
                    self.shoulder_base_angle,
                    self.sweep_amplitude,
                    self.sweep_angular_freq,
                    self.sim_time_array,
                ],
                outputs=[self.live_joint_q],
                device=self.solver.device,
            )
            newton.eval_fk(
                self.model,
                self.live_joint_q,
                self.model.joint_qd,
                self.state_0,
                indices=self.arm_articulation_indices,
            )
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.launch(
                track_hand_kernel,
                dim=1,
                inputs=[self.hand, self.cable_bodies[-1], self.hand_end_local],
                outputs=[self.state_0.body_q, self.state_1.body_q],
                device=self.solver.device,
            )
            wp.launch(
                advance_time_kernel,
                dim=1,
                inputs=[self.sim_time_array, self.sim_dt],
                device=self.solver.device,
            )

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

    def test_final(self):
        """Test cable-swings-weight simulation for stability (called after simulation)."""
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"
            assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"

            # The hand-side cable end should stay welded to the hand.
            hand_pos = body_positions[self.hand][:3]
            hand_end_pos = body_positions[self.cable_bodies[-1]][:3]
            assert np.linalg.norm(hand_end_pos - hand_pos) < 0.05, (
                f"Hand-side cable end detached from hand: {hand_end_pos} vs {hand_pos}"
            )

            # The weight should be above the ground and not have flown off
            # to an unreasonable distance.
            weight_pos = body_positions[self.weight][:3]
            assert weight_pos[2] > -0.1, f"Weight fell through the ground: {weight_pos}"
            assert np.linalg.norm(weight_pos - hand_pos) < 5.0, f"Weight flew unreasonably far away: {weight_pos}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
