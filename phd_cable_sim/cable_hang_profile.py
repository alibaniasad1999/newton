# PhD Cable Project - Newton simulation scripts
#
# Hang a Newton VBD cable between two given attachment points and export its
# settled 2-D profile in the formats consumed by the IsaacLab-repo image
# pipeline (scripts/cable_simulation/image_utils/extract_cable_profile.py),
# so a real cable photographed with that tool can be compared directly
# against Newton -- the same way it is compared against the IsaacLab
# capsule-chain / FEM / warp-rod sims.
#
# Inputs (the same things you measure on the real cable):
#   --p1 X,Z     first attachment point [m] (the ANCHOR; the image tool uses
#                the anchor as x=0, so keep p1 at 0,<height> for comparisons)
#   --p2 X,Z     second attachment point [m]
#   --p3 X,Z     OPTIONAL third attachment point [m]. When given, the cable
#                runs p1 -> p2 -> p3 and is held at all three: the ends are
#                clamped and p2 grabs the cable through a ball joint to the
#                world -- a pinch/hook that holds the point but lets the
#                cable pivot, giving two independent sagging spans.
#   --mid-fraction f  arc-length fraction of the cable held at p2 (3-point
#                mode only); default splits the length across the two spans
#                proportionally to their chord distances.
#   --length L   cable length [m] (must exceed the summed anchor distances
#                to sag)
# plus the physical cable parameters, defaulting to the shared TPU target in
# IsaacLab's cable_config.py (E=40 MPa, r=1.5 mm, rho=1150 kg/m^3). Young's
# modulus is converted to Newton's per-joint constants as
#   k_stretch = EA / L_seg [N/m],   k_bend = EI / L_seg [N*m/rad],
# with EA = E*pi*r^2 and EI = E*pi*r^4/4.
#
# Outputs (written next to --out prefix):
#   <out>_trajectory.csv  per-frame node positions, columns t, nXX_x, nXX_z
#                         (matches load_sim_profile in the image tool)
#   <out>_profile.csv     settled shape, columns x_m, z_m (CableProfile format)
#
# Compare against a real photo profile with the EXISTING IsaacLab tool:
#   python extract_cable_profile.py compare \
#       --real results/IMG_0501/profile.csv \
#       --sim  <out>_trajectory.csv --plot cmp.png
#
# Run with the GUI:
#   uv run -m phd_cable_sim.cable_hang_profile --viewer gl --p1 0,1.0 --p2 0.8,1.0 --length 1.0
#
# Three points (cable held up in the middle at p2):
#   uv run -m phd_cable_sim.cable_hang_profile --viewer gl \
#       --p1 0,1.0 --p2 0.5,1.2 --p3 1.0,1.0 --length 1.4
#
# Run headless with the built-in sanity checks (includes an analytic
# catenary cross-check for equal-height anchors):
#   uv run -m phd_cable_sim.cable_hang_profile --viewer null --test --num-frames 480

import csv
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.math import quat_between_vectors_robust


def parse_point(text: str) -> tuple[float, float]:
    x, z = (float(v) for v in text.split(","))
    return x, z


def sagged_polyline(p1: np.ndarray, p2: np.ndarray, length: float, num_segments: int) -> np.ndarray:
    """Initial cable centerline: a smooth sagging curve from p1 to p2 in the
    x-z plane whose POLYLINE arc length equals `length` (bisection on the
    sag amplitude). The rod's joint rest lengths come from these positions,
    so the polyline length must match the requested cable length exactly.
    """
    t = np.linspace(0.0, 1.0, num_segments + 1)
    straight = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
    chord = float(np.linalg.norm(p2 - p1))
    if length <= chord:
        print(f"[warn] length {length:.3f} m <= anchor distance {chord:.3f} m: cable starts taut")
        return straight

    def polyline(sag: float) -> np.ndarray:
        pts = straight.copy()
        pts[:, 2] -= sag * np.sin(np.pi * t)  # sag straight down (-z)
        return pts

    def arc_length(pts: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    lo, hi = 0.0, length  # arc_length(hi) > length always
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if arc_length(polyline(mid)) < length:
            lo = mid
        else:
            hi = mid
    return polyline(0.5 * (lo + hi))


def analytic_catenary_z(x: np.ndarray, x1: float, x2: float, z_anchor: float, length: float) -> np.ndarray | None:
    """Analytic catenary through two EQUAL-HEIGHT anchors at (x1, z_anchor)
    and (x2, z_anchor) with the given arc length; None if no sag."""
    d = abs(x2 - x1)
    if length <= d:
        return None
    # Solve L = 2 a sinh(d / (2a)) for a by bisection.
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        a = math.sqrt(lo * hi)
        if 2.0 * a * math.sinh(d / (2.0 * a)) > length:
            lo = a
        else:
            hi = a
    a = math.sqrt(lo * hi)
    xc = 0.5 * (x1 + x2)
    return z_anchor + a * (np.cosh((x - xc) / a) - math.cosh(d / (2.0 * a)))


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        # The cable must live in the world x-z plane: the CSV exports and the
        # catenary test read x as the horizontal coordinate (the image tool's
        # convention). For a face-on view the CAMERA is rotated instead (see
        # the set_camera call below) -- do not swap the cable onto another
        # plane to fix the view, that zeroes the exported x column.
        def to3d(text: str) -> np.ndarray:
            x, z = parse_point(text)
            return np.array([x, 0.0, z])

        # 2 or 3 attachment points, in cable order.
        self.anchors = [to3d(args.p1), to3d(args.p2)]
        if args.p3 is not None:
            self.anchors.append(to3d(args.p3))
        self.p1 = self.anchors[0]
        self.p2 = self.anchors[-1]
        self.cable_length = float(args.length)
        cable_radius = float(args.radius)
        young_modulus = float(args.young_modulus)
        density = float(args.density)

        # Simulation cadence (matches the other scripts in this project)
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Discretization and per-joint material constants from E.
        self.num_elements = int(args.segments)
        seg_len = self.cable_length / self.num_elements
        if args.site_params:
            # The literal values from NVIDIA's Newton manipulation blog post,
            # passed RAW per joint exactly as the snippet shows -- NO EA/L_seg
            # normalization, so the effective cable depends on --segments.
            # This models THEIR stiff industrial harness (EI=3 N*m^2 is
            # ~20,000x stiffer in bending than the TPU cable), for
            # side-by-side comparison with the physical default below.
            cable_radius = 0.003
            stretch_stiffness = 1.0e12
            bend_stiffness = 3.0
            stretch_damping = 1.0e-3
            bend_damping = 1.0
            print(
                "[cable] SITE PARAMS (raw per-joint, blog values): r=3.0 mm, "
                "k_stretch=1e12, k_bend=3.0, damping=(1e-3, 1.0)"
            )
        else:
            area = math.pi * cable_radius**2
            second_moment = math.pi * cable_radius**4 / 4.0
            ea = young_modulus * area  # [N]
            ei = young_modulus * second_moment  # [N*m^2]
            stretch_stiffness = ea / seg_len
            bend_stiffness = ei / seg_len
            stretch_damping = 1.0e-4
            bend_damping = 0.5 * bend_stiffness  # heavy relative damping: we want the static shape
            print(
                f"[cable] L={self.cable_length} m, r={cable_radius * 1000:.1f} mm, E={young_modulus / 1e6:.0f} MPa -> "
                f"EA={ea:.1f} N, EI={ei:.2e} N*m^2, per-joint k_stretch={stretch_stiffness:.2e} N/m, "
                f"k_bend={bend_stiffness:.2e} N*m/rad, mass={density * area * self.cable_length * 1000:.1f} g"
            )

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0

        cable_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=density,
            ke=builder.default_shape_cfg.ke,
            kd=builder.default_shape_cfg.kd,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=builder.default_shape_cfg.mu,
            restitution=builder.default_shape_cfg.restitution,
        )

        # Initial sagging centerline with the exact requested arc length,
        # one sagging span per anchor pair, and per-segment quaternions
        # aligning local +Z to each segment. Span arc lengths split the
        # total length by --mid-fraction (default: proportional to chords).
        chords = [float(np.linalg.norm(b - a)) for a, b in zip(self.anchors[:-1], self.anchors[1:], strict=True)]
        if len(self.anchors) == 3:
            f = args.mid_fraction if args.mid_fraction is not None else chords[0] / sum(chords)
            assert 0.0 < f < 1.0, f"--mid-fraction must be in (0, 1), got {f}"
            self.span_lengths = [f * self.cable_length, (1.0 - f) * self.cable_length]
            n1 = min(max(int(round(f * self.num_elements)), 2), self.num_elements - 2)
            span_segments = [n1, self.num_elements - n1]
        else:
            self.span_lengths = [self.cable_length]
            span_segments = [self.num_elements]

        span_pts = [
            sagged_polyline(self.anchors[i], self.anchors[i + 1], self.span_lengths[i], span_segments[i])
            for i in range(len(self.span_lengths))
        ]
        # Concatenate spans into one continuous polyline (shared node at p2).
        pts = np.concatenate([span_pts[0]] + [sp[1:] for sp in span_pts[1:]], axis=0)
        # Node index held by the middle attachment (3-point mode).
        self.span_segments = span_segments
        self.mid_node = span_segments[0] if len(self.anchors) == 3 else None
        positions = [wp.vec3(*p) for p in pts]
        # Per-segment chord lengths: the sagged polyline is parameter-uniform,
        # not length-uniform, and node_positions() must use the real lengths.
        self.segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        quaternions = []
        for i in range(self.num_elements):
            d = pts[i + 1] - pts[i]
            d = d / np.linalg.norm(d)
            quaternions.append(quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*d)))

        cable_bodies, _cable_joints = builder.add_rod(
            positions=positions,
            quaternions=quaternions,
            radius=cable_radius,
            cfg=cable_shape_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=stretch_damping,
            bend_stiffness=bend_stiffness,
            bend_damping=bend_damping,
            label="hang_cable",
            body_frame_origin="com",
        )
        self.cable_bodies = cable_bodies

        # Pin both end segments (zero mass/inertia): the verified kinematic
        # anchor pattern used across this project. Their endpoints sit
        # exactly at the first/last anchors by construction.
        for body in (cable_bodies[0], cable_bodies[-1]):
            builder.body_mass[body] = 0.0
            builder.body_inv_mass[body] = 0.0
            builder.body_inertia[body] = wp.mat33(0.0)
            builder.body_inv_inertia[body] = wp.mat33(0.0)

        # Middle attachment (3-point mode): a ball joint from the WORLD to
        # the cable node at p2 -- a pinch/hook that holds the point but lets
        # the cable rotate through it, so each side sags independently. Kept
        # out of any articulation (the rod already parents this body via its
        # cable joint); newton.eval_fk skips it, SolverVBD enforces it.
        if self.mid_node is not None:
            mid_body = cable_bodies[self.mid_node]
            builder.add_joint_ball(
                parent=-1,
                child=mid_body,
                parent_xform=wp.transform(wp.vec3(*self.anchors[1]), wp.quat_identity()),
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, -0.5 * self.segment_lengths[self.mid_node]), wp.quat_identity()
                ),
            )

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

        self.trajectory_rows: list[list[float]] = []

        self.viewer.set_model(self.model)
        # Face-on view of the x-z cable plane: camera pulled back along -Y,
        # looking along +Y (Z-up yaw convention: front = (cos yaw, sin yaw)).
        # --camera-distance overrides how far back it sits; smaller = nearer.
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        span = float(np.linalg.norm(self.p2 - self.p1))
        cam_dist = args.camera_distance if args.camera_distance is not None else max(1.5 * span, 0.75)
        self.viewer.set_camera(pos=wp.vec3(center[0], center[1] - cam_dist, center[2]), pitch=0.0, yaw=90.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            camera.set_pivot(wp.vec3(*center))

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def node_positions(self) -> np.ndarray:
        """Cable centerline nodes (num_elements + 1, 3): segment endpoints."""
        body_q = self.state_0.body_q.numpy()
        pts = np.empty((len(self.cable_bodies) + 1, 3))
        for i, body in enumerate(self.cable_bodies):
            half = 0.5 * self.segment_lengths[i]
            p = body_q[body, :3]
            x, y, z, w = body_q[body, 3:]
            z_axis = np.array([2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)])
            pts[i] = p - half * z_axis
            if i == len(self.cable_bodies) - 1:
                pts[i + 1] = p + half * z_axis
        return pts

    def step(self):
        """Advance simulation by one frame and record the trajectory row."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        pts = self.node_positions()
        self.trajectory_rows.append([self.sim_time, *pts[:, 0].tolist(), *pts[:, 2].tolist()])

    def render(self):
        """Render the current simulation state to the viewer."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def write_outputs(self):
        """Write <out>_trajectory.csv (image-tool sim format) and
        <out>_profile.csv (image-tool CableProfile format)."""
        n = len(self.cable_bodies) + 1
        node_names = [f"n{i:02d}" for i in range(n)]
        traj_path = f"{self.args.out}_trajectory.csv"
        with open(traj_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + [f"{nn}_x" for nn in node_names] + [f"{nn}_z" for nn in node_names])
            for row in self.trajectory_rows:
                w.writerow([f"{v:.6f}" for v in row])

        pts = self.node_positions()
        profile_path = f"{self.args.out}_profile.csv"
        with open(profile_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_m", "z_m"])
            for p in pts:
                w.writerow([f"{p[0]:.6f}", f"{p[2]:.6f}"])

        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        sag = float(np.min(pts[:, 2]))
        vel = np.abs(self.state_0.body_qd.numpy()).max()
        print(f"[out] {traj_path}  ({len(self.trajectory_rows)} frames)")
        print(f"[out] {profile_path}")
        print(
            f"[result] settled arc length {arc:.4f} m (target {self.cable_length:.4f}), lowest z {sag:.4f} m, max |v| {vel:.2e}"
        )

    def test_final(self):
        """Stability + shape checks, incl. an analytic-catenary cross-check."""
        pts = self.node_positions()
        body_velocities = self.state_0.body_qd.numpy()
        assert np.isfinite(pts).all(), "Non-finite positions"
        assert np.isfinite(body_velocities).all(), "Non-finite velocities"
        assert (np.abs(body_velocities) < 1.0).all(), "Cable did not settle (velocities too large)"

        # Endpoints stay pinned; the middle attachment (if any) holds its
        # point through the ball joint (allow a little solver slack).
        assert np.linalg.norm(pts[0] - self.anchors[0]) < 1e-3, f"End 1 detached: {pts[0]} vs {self.anchors[0]}"
        assert np.linalg.norm(pts[-1] - self.anchors[-1]) < 1e-3, f"End 2 detached: {pts[-1]} vs {self.anchors[-1]}"
        if self.mid_node is not None:
            drift = np.linalg.norm(pts[self.mid_node] - self.anchors[1])
            assert drift < 5e-3, f"Middle attachment slipped {drift * 1000:.1f} mm from {self.anchors[1]}"

        # Arc length preserved (near-inextensible).
        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        assert abs(arc - self.cable_length) < 0.02 * self.cable_length, (
            f"Arc length drifted: {arc:.4f} vs {self.cable_length:.4f}"
        )

        # Each span between equal-height attachments must settle onto the
        # analytic catenary for its own span arc length (the ball-joint
        # pinch holds a material point, so no length migrates between spans).
        node = 0
        for i, (n_seg, span_len) in enumerate(zip(self.span_segments, self.span_lengths, strict=True)):
            a, b = self.anchors[i], self.anchors[i + 1]
            span_nodes = pts[node : node + n_seg + 1]
            node += n_seg
            if abs(a[2] - b[2]) > 1e-9:
                continue
            z_ref = analytic_catenary_z(span_nodes[:, 0], a[0], b[0], a[2], span_len)
            if z_ref is None:
                continue
            rms = float(np.sqrt(np.mean((span_nodes[:, 2] - z_ref) ** 2)))
            print(f"[test] span {i + 1} RMS vs analytic catenary: {rms * 1000:.2f} mm")
            if self.args.site_params:
                # A stiff harness is NOT a catenary -- report, don't assert.
                print("[test] (catenary assertion skipped: --site-params models a stiff rod)")
                continue
            assert rms < 0.01, f"Span {i + 1} deviates from catenary: RMS {rms * 1000:.1f} mm"

        self.write_outputs()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--p1", type=str, default="0.0,1.0", help="Anchor 1 'X,Z' [m] (use as origin).")
        parser.add_argument("--p2", type=str, default="0.8,1.0", help="Anchor 2 'X,Z' [m].")
        parser.add_argument(
            "--p3",
            type=str,
            default=None,
            help="Optional anchor 3 'X,Z' [m]; cable then runs p1 -> p2 -> p3, held at all three.",
        )
        parser.add_argument(
            "--mid-fraction",
            type=float,
            default=None,
            help="3-point mode: arc-length fraction of the cable held at p2 (default: chord-proportional).",
        )
        parser.add_argument("--length", type=float, default=1.0, help="Cable length [m].")
        parser.add_argument("--segments", type=int, default=100, help="Number of rod segments.")
        parser.add_argument("--radius", type=float, default=1.5e-3, help="Cable radius [m] (IsaacLab TPU default).")
        parser.add_argument("--young-modulus", type=float, default=40e6, help="Young's modulus [Pa] (TPU default).")
        parser.add_argument("--density", type=float, default=1150.0, help="Density [kg/m^3] (TPU default).")
        parser.add_argument("--out", type=str, default="cable_hang", help="Output file prefix.")
        parser.add_argument(
            "--camera-distance",
            type=float,
            default=None,
            help="Camera distance from the cable [m]; smaller = closer. Default frames the whole cable.",
        )
        parser.add_argument(
            "--site-params",
            action="store_true",
            help="Use the raw NVIDIA-blog rod values (r=3 mm, stretch 1e12, bend 3.0, per joint, "
            "no length normalization) instead of the physical EA/EI conversion -- a stiff "
            "industrial harness, for comparison.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=480)  # 8 s: plenty to settle
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if not (args is not None and args.test):
        example.write_outputs()
