# PhD Cable Project - Newton simulation scripts
#
# A single Franka Panda (FR3) arm, held in a fixed reach-forward pose (every
# arm link's mass/inertia is zeroed, matching the kinematic-anchor pattern
# used elsewhere in this project), grips one end of a charger-cable-like
# cable in its hand. The cable is long enough that its free end reaches
# past the hand's height and settles onto the ground plane under gravity
# and ground contact, draping naturally -- no anchoring/tension hack is
# needed for the free end here (unlike franka_cable_hold.py's two-hands
# case), since the ground itself is what the free end rests against.
#
# Mouse controls (GL viewer): see phd_cable_sim/cube_anchor_cable_swing.py
#
# Run with the GUI:
#   uv run -m phd_cable_sim.franka_cable_floor_drape --viewer gl
#
# Run headless with the built-in sanity checks:
#   uv run -m phd_cable_sim.franka_cable_floor_drape --viewer null --test --num-frames 120

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

        # Charger-cable-like properties: thin, light, floppy. Long enough
        # (relative to the hand's height above the ground) that the free
        # end reaches past the floor and drapes/coils rather than dangling
        # in mid-air.
        self.num_elements = 48
        self.cable_length = 1.4
        cable_radius = 0.003  # 3 mm
        stretch_stiffness = 1.0e5
        stretch_damping = 1.0e-4
        bend_stiffness = 1.0e-2
        bend_damping = 1.0e-2

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

        # Arm base positioned so the hand sits comfortably above the ground
        # with room for the cable to reach down and drape past it.
        arm_base_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        hand_xform = wp.transform_multiply(arm_base_xform, hand_local)
        hand_p = wp.transform_get_translation(hand_xform)

        arm_body_start = builder.body_count
        arm_shape_start = builder.shape_count
        arm_sub = newton.ModelBuilder()
        arm_sub.add_urdf(urdf_path, floating=False)
        arm_sub.joint_q = list(FRANKA_POSE_Q)
        builder.add_builder(arm_sub, xform=arm_base_xform, label_prefix="arm")
        arm_body_end = builder.body_count
        arm_shape_end = builder.shape_count
        hand = _find_body(builder.body_label, "arm/fr3/fr3_hand_tcp", arm_body_start, arm_body_end)

        # Hold the whole arm fixed: no arm motion here, just a static
        # cable-holding pose (see franka_cable_hold.py for a version where
        # one arm's shoulder joint is animated).
        for body in range(arm_body_start, arm_body_end):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        # Disable collision on the (permanently frozen) arm meshes: this was
        # the dominant simulation cost in franka_cable_hold.py (~34 ms/substep
        # of wasted mesh-vs-mesh narrow phase across ~65 Franka collision
        # shapes for one arm). The cable still collides with the ground.
        for shape in range(arm_shape_start, arm_shape_end):
            builder.shape_collision_group[shape] = 0

        # Cable hangs straight down from the hand, long enough that its
        # free end reaches well past the ground plane -- the excess length
        # settles into a coil/drape on the floor.
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=hand_p,
            direction=wp.vec3(0.0, 0.0, -1.0),
            length=self.cable_length,
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
        self.hand = hand

        # Pin only the top (gripped) segment to the hand -- zero mass/inertia
        # matches the fixed hand it's attached to. The free (bottom) end is
        # left as a normal cable segment: no anchor, no pinning. It settles
        # onto the ground under gravity and ground contact, which is what
        # actually holds it in place once it lands -- unlike
        # franka_cable_hold.py's suspended-in-air case, there's no need for
        # a soft-anchor hack here.
        top_body = cable_bodies[0]
        builder.body_mass[top_body] = 0.0
        builder.body_inv_mass[top_body] = 0.0
        builder.body_inertia[top_body] = wp.mat33(0.0)
        builder.body_inv_inertia[top_body] = wp.mat33(0.0)

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

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.viewer.set_model(self.model)

        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            ps = picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 20.0
            ps[0]["pick_damping"] = 1.0
            picking.pick_state.assign(ps)

        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.set_pivot(wp.vec3(hand_p[0], hand_p[1], 0.3))

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
        """Test the cable-drape simulation for stability (called after simulation)."""
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"
            assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"

            # The gripped top segment should stay fixed at the hand.
            top_pos = body_positions[self.cable_bodies[0]][:3]
            hand_pos = body_positions[self.hand][:3]
            assert np.linalg.norm(top_pos - hand_pos) < 0.1, (
                f"Gripped cable end detached from hand: {top_pos} vs hand {hand_pos}"
            )

            # The free end should have settled down near the ground, not be
            # left dangling in mid-air or clipped through the floor.
            free_end_z = body_positions[self.cable_bodies[-1]][2]
            assert -0.05 < free_end_z < 0.3, f"Free cable end not resting near the ground: z={free_end_z:.3f}"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
