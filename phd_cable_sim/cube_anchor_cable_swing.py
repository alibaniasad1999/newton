# PhD Cable Project - Newton simulation scripts
#
# A 10 cm cube anchor oscillates sinusoidally along X, 2 m above the ground
# (Isaac-Sim-style driven prim). A 1 m charger-cable-like cable (thin,
# light, low bend stiffness) hangs from the bottom face of the cube and
# follows its motion, letting us watch the cable swing/whip in response to
# the anchor's driven oscillation.
#
# Mouse controls (GL viewer):
#   Right-click + drag  - grab and move any body (including the cable)
#   Left-click + drag   - rotate the view
#   Middle-click + drag - pan / orbit / dolly
#   Scroll wheel         - zoom (dolly); Ctrl + Scroll - adjust FOV
#
# Run with the GUI:
#   uv run -m phd_cable_sim.cube_anchor_cable_swing --viewer gl
#
# Run headless with the built-in sanity checks:
#   uv run -m phd_cable_sim.cube_anchor_cable_swing --viewer null --test --num-frames 120

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def drive_anchor_and_cable_top_kernel(
    anchor_body: int,
    cable_top_body: int,
    center: wp.vec3,
    amplitude: float,
    angular_freq: float,
    rot: wp.quat,
    cable_top_local_z: float,
    t: wp.array[float],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Drive the anchor cube along a sine wave in X and keep the cable's
    pinned top segment rigidly following the cube's bottom face."""
    cur_t = t[0]
    x = center[0] + amplitude * wp.sin(angular_freq * cur_t)

    anchor_pos = wp.vec3(x, center[1], center[2])
    anchor_xform = wp.transform(anchor_pos, rot)
    body_q0[anchor_body] = anchor_xform
    body_q1[anchor_body] = anchor_xform

    cable_top_pos = anchor_pos + wp.vec3(0.0, 0.0, cable_top_local_z)
    cable_top_xform = wp.transform(cable_top_pos, rot)
    body_q0[cable_top_body] = cable_top_xform
    body_q1[cable_top_body] = cable_top_xform


@wp.kernel
def advance_time_kernel(t: wp.array[float], dt: float):
    t[0] = t[0] + dt


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

        # Anchor cube: 10 cm per side, centered 2 m up, driven along a sine
        # wave in X.
        cube_half_extent = 0.05
        anchor_height = 2.0
        self.anchor_center = wp.vec3(0.0, 0.0, anchor_height)
        self.anchor_amplitude = 0.3  # [m]
        self.anchor_angular_freq = 2.0 * np.pi * 0.5  # 0.5 Hz
        anchor_pos = self.anchor_center

        # Charger-cable-like properties: thin, light, floppy
        self.num_elements = 24
        self.cable_length = 1.0
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

        # Kinematic anchor cube: a real body (not body=-1 static geometry) so
        # it can be driven each substep. Its motion is written directly to
        # body_q rather than joined to the cable via add_joint_fixed: coupling
        # the cable's first segment to a second body through a joint makes
        # SolverVBD treat the whole chain as one rigid articulation, which
        # (as tested) instantly damps out any velocity on the free end.
        # Instead, both the cube and the cable's pinned top segment are
        # driven directly and independently, keeping the cable's own
        # stretch/bend dynamics intact.
        anchor = builder.add_body(xform=wp.transform(anchor_pos, wp.quat_identity()), is_kinematic=True, label="anchor_cube")
        builder.add_shape_box(
            anchor,
            hx=cube_half_extent,
            hy=cube_half_extent,
            hz=cube_half_extent,
            label="anchor_cube_shape",
        )
        self.anchor_body = anchor

        # Straight cable hanging down from the bottom face of the anchor cube
        attach_point = anchor_pos - wp.vec3(0.0, 0.0, cube_half_extent)
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=attach_point,
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
        segment_length = self.cable_length / self.num_elements
        # Local Z offset (relative to the cube center) of the cable's pinned
        # top segment body origin; body_frame_origin="com" places it at the
        # segment midpoint, half a segment below the cube's bottom face.
        self.cable_top_local_z = -cube_half_extent - 0.5 * segment_length

        # Pin the top segment (zero mass/inertia) so the solver doesn't try
        # to move it under cable tension; its position is driven directly
        # each substep to track the cube (see drive_anchor_and_cable_top_kernel).
        top_body = cable_bodies[0]
        builder.body_mass[top_body] = 0.0
        builder.body_inv_mass[top_body] = 0.0
        builder.body_inertia[top_body] = wp.mat33(0.0)
        builder.body_inv_inertia[top_body] = wp.mat33(0.0)
        self.cable_top_body = top_body

        builder.add_ground_plane()
        builder.color()

        self.model = builder.finalize()
        self.contacts = self.model.contacts()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.sim_iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        self.viewer.set_model(self.model)

        # Make it easy to grab the cable with the mouse (right-click drag):
        # a lower stiffness/damping than the default feels better for a
        # light, floppy cable than for a rigid object.
        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            ps = picking.pick_state.numpy()
            ps[0]["pick_stiffness"] = 20.0
            ps[0]["pick_damping"] = 1.0
            picking.pick_state.assign(ps)

        # Face-on view of the x-z plane the cable swings in (Z-up yaw
        # convention: front = (cos yaw, sin yaw)), pivoting on the cable so
        # the zoom limit (a fixed minimum distance from the pivot) doesn't
        # stop you from getting close to this small, ~1 m long cable.
        center = wp.vec3(0.0, 0.0, anchor_height - 0.5)
        self.viewer.set_camera(pos=wp.vec3(center[0], center[1] - 2.0, center[2]), pitch=0.0, yaw=90.0)
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

            wp.launch(
                drive_anchor_and_cable_top_kernel,
                dim=1,
                inputs=[
                    self.anchor_body,
                    self.cable_top_body,
                    self.anchor_center,
                    self.anchor_amplitude,
                    self.anchor_angular_freq,
                    wp.quat_identity(),
                    self.cable_top_local_z,
                    self.sim_time_array,
                ],
                outputs=[self.state_0.body_q, self.state_1.body_q],
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

    def test_final(self):
        """Test cable swing simulation for stability (called after simulation)."""
        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            assert np.isfinite(body_positions).all(), "Non-finite positions"
            assert np.isfinite(body_velocities).all(), "Non-finite velocities"
            assert (np.abs(body_velocities) < 5e2).all(), "Velocities too large"

            # The anchor cube should stay on its driven sine-wave path in X,
            # centered at the configured height, within the driven amplitude.
            anchor_pos = body_positions[self.anchor_body][:3]
            center = np.array([self.anchor_center[0], self.anchor_center[1], self.anchor_center[2]])
            assert abs(anchor_pos[0] - center[0]) <= self.anchor_amplitude + 1e-3, (
                f"Anchor cube X went outside driven amplitude: {anchor_pos[0]}"
            )
            assert np.linalg.norm(anchor_pos[1:] - center[1:]) < 1e-3, (
                f"Anchor cube moved off its Y/Z track: {anchor_pos}"
            )

            # The pinned cable top segment should stay locked to the cube's
            # bottom face (fixed local offset), not drift independently.
            top_pos = body_positions[self.cable_bodies[0]][:3]
            expected_top_pos = anchor_pos + np.array([0.0, 0.0, self.cable_top_local_z])
            assert np.linalg.norm(top_pos - expected_top_pos) < 1e-3, (
                f"Cable top segment detached from anchor cube: {top_pos} (expected {expected_top_pos})"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
