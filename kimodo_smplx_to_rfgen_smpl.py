#!/usr/bin/env python3
"""
kimodo_smplx_to_rfgen_smpl.py

Converts a Kimodo-SMPLX-RP AMASS-style NPZ into the SMPL NPZ format that
RF-Genesis (`obj_diff.npz`) expects.

Pipeline:
  kimodo --model Kimodo-SMPLX-RP --text "..." --output kimodo_out.npz
  python kimodo_smplx_to_rfgen_smpl.py kimodo_out.npz \
         RF-Genesis/output/<name>/obj_diff.npz
  cd RF-Genesis && python run.py -o "ignored" -e "..." -n <name>
                                  ^ Step 1 (MDM) is auto-skipped because
                                    obj_diff.npz already exists.

Three things this adapter handles:
  (a) Joint count   : SMPL-X body (66-dim, 22 joints) -> SMPL (72-dim, 24 joints).
                      The two SMPL hand joints (22, 23) are zero-padded; finger
                      articulation is invisible at AWR1843 mmWave resolution.
  (b) Coordinates   : AMASS (z-up, +y forward) -> SMPL/y-up (y-up, -z forward).
                      A -90 deg rotation about x is applied to root_orient and
                      trans. pose_body (parent-relative axis-angles) is
                      invariant under a world-frame rotation and passes through
                      unchanged.
  (c) Frame rate    : Resampled to the 30 fps that RFGen's signal_generator
                      hardcodes. Translation: linear interpolation. Rotations:
                      per-joint Slerp (scipy.spatial.transform.Slerp).
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


# AMASS (z-up, +y forward) -> SMPL y-up (y-up, -z forward) is a -90 deg
# rotation about the x-axis. det = +1 (proper rotation, no chirality flip).
#   e_y_amass (forward) ->  -e_z_yup
#   e_z_amass (up)      ->  +e_y_yup
AMASS_TO_YUP = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0, -1.0,  0.0],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# (c) Frame-rate resampling
# ---------------------------------------------------------------------------
def _slerp_axis_angle(aa, t_in, t_out):
    """Slerp interpolation for an axis-angle sequence shaped (T, 3)."""
    rots = Rotation.from_rotvec(aa)
    slerp = Slerp(t_in, rots)
    return slerp(t_out).as_rotvec()


def resample_motion(root_orient, pose_body, trans, fps_in, fps_out):
    T = root_orient.shape[0]
    if T < 2 or abs(fps_in - fps_out) < 1e-6:
        return root_orient, pose_body, trans

    duration = (T - 1) / float(fps_in)
    T_new = max(2, int(round(duration * fps_out)) + 1)
    t_in  = np.linspace(0.0, duration, T)
    t_out = np.linspace(0.0, duration, T_new)

    trans_out = interp1d(t_in, trans, axis=0, kind='linear')(t_out)

    root_out = _slerp_axis_angle(root_orient, t_in, t_out)

    body = pose_body.reshape(T, -1, 3)                      # (T, 21, 3)
    n_joints = body.shape[1]
    body_out = np.zeros((T_new, n_joints, 3), dtype=body.dtype)
    for j in range(n_joints):
        body_out[:, j, :] = _slerp_axis_angle(body[:, j, :], t_in, t_out)
    body_out = body_out.reshape(T_new, -1)

    return (root_out.astype(np.float32),
            body_out.astype(np.float32),
            trans_out.astype(np.float32))


# ---------------------------------------------------------------------------
# (b) Coordinate-system change
# ---------------------------------------------------------------------------
def amass_to_yup(root_orient, trans, M=AMASS_TO_YUP):
    """Apply AMASS->y-up to GLOBAL root_orient and translation only.

    pose_body (parent-relative joint rotations) is invariant under a global
    coordinate change and is left untouched.

    Rotation: R_yup = M @ R_amass (LEFT multiplication, NOT conjugation).

    Why left multiplication, not conjugation?

      SMPL_Layer in RFGen produces vertices in SMPL's NATIVE canonical frame
      (a fixed orientation regardless of which world frame we ultimately
      render in). RFGen then passes those vertices straight to Mitsuba, which
      renders y-up. So R_root must rotate the SMPL canonical onto the body's
      pose AS EXPRESSED IN Y-UP -- it does not transform a rotation between
      coord systems.

      AMASS root_orient rotates the SMPL canonical onto the body's pose AS
      EXPRESSED IN AMASS Z-UP. So we need to compose with the z-up -> y-up
      world rotation:  R_yup = M @ R_amass.

      Conjugation (M @ R @ M^T) is the right transform for changing the
      *coordinate frame* a rotation is expressed in (e.g., expressing the
      same physical rotation in a different basis), but here we are not
      changing the basis of R -- we are composing R with an additional
      world rotation. Using conjugation here makes an upright AMASS body
      appear lying down in RFGen's render (head pointing along z instead
      of y), which is exactly the symptom of the original bug.

    Translation: t_yup = M @ t_amass  (positions transform by the same M).
    """
    # Translation: t_yup = M @ t_amass (per row)
    trans_yup = trans @ M.T

    # Rotation: R_yup = M @ R_amass  -- left multiplication, NOT conjugation.
    R_amass = Rotation.from_rotvec(root_orient).as_matrix()    # (T, 3, 3)
    R_yup   = np.einsum('ij,tjk->tik', M, R_amass)             # (T, 3, 3)
    root_yup = Rotation.from_matrix(R_yup).as_rotvec()         # (T, 3)

    return root_yup.astype(np.float32), trans_yup.astype(np.float32)


# ---------------------------------------------------------------------------
# (a) SMPL-X -> SMPL pose layout
# ---------------------------------------------------------------------------
def build_smpl_pose72(root_orient, pose_body):
    """SMPL pose72 = [root(3) | body 21 joints (63) | L_hand(3) | R_hand(3)] = 72.
    SMPL-X body shares its first 22 joints (root + 21) with SMPL exactly.
    SMPL joints 22 and 23 (L_hand, R_hand) are zero-padded; mmWave radar at
    AWR1843 resolution (~3.75 cm range, deg-scale angle) cannot resolve
    finger articulation, so this is a lossless approximation in practice."""
    T = root_orient.shape[0]
    pose72 = np.zeros((T, 72), dtype=np.float32)
    pose72[:, 0:3]   = root_orient
    pose72[:, 3:66]  = pose_body
    # pose72[:, 66:72] = 0   (L_hand, R_hand)  -- already zero
    return pose72


# ---------------------------------------------------------------------------
# (d) Safe tail-padding to work around RFGen signal_generator.py off-by-one
# ---------------------------------------------------------------------------
def _read_radar_config(explicit_path, output_npz_path):
    """Return (frame_per_second, chirp_per_frame) from a radar config JSON.

    Search order:
      1. --radar-config  (if given)
      2. <output_dir>/../../models/TI1843_config.json   (typical RFGen layout)
      3. ./models/TI1843_config.json
      4. Built-in TI1843 defaults
    """
    import json
    candidates = []
    if explicit_path:
        candidates.append(os.path.abspath(explicit_path))
    out_dir = os.path.dirname(os.path.abspath(output_npz_path))
    # Walk up two levels: output/<name>/obj_diff.npz -> RF-Genesis/
    candidates.append(os.path.normpath(os.path.join(out_dir, "..", "..",
                                                    "models", "TI1843_config.json")))
    candidates.append(os.path.abspath(os.path.join("models", "TI1843_config.json")))

    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r") as f:
                cfg = json.load(f)
            return (int(cfg["frame_per_second"]),
                    int(cfg["chirp_per_frame"]),
                    path)

    # TI1843 defaults
    return 10, 128, None


def compute_safe_pad(N, motion_fps=30, radar_fps=10, chirp_per_frame=128, min_pad=2):
    """Smallest k >= min_pad such that RFGen's signal_generator.py interpolator
    cannot trigger an out-of-bounds `frames[frame_index + 1]` for any radar
    chirp query.

    Bug condition derived from RFGen source:
        max_query_time   = (radar_count - 1/chirp_per_frame) / radar_fps
        radar_count      = floor((N+k) * radar_fps / motion_fps)
        frame_index      = int(max_query_time * motion_fps)
        bug iff frame_index + 1 >= num_motion_frames (= N+k)

    Algebraic simplification gives the safe condition:
        ((N+k) * radar_fps) mod motion_fps  >=  radar_fps - motion_fps/chirp_per_frame
    """
    threshold = radar_fps - motion_fps / chirp_per_frame
    # Search a small window; for any reasonable rates a safe k exists within
    # one motion-fps-period. We bound the search at motion_fps + min_pad.
    for k in range(min_pad, min_pad + motion_fps + 1):
        if ((N + k) * radar_fps) % motion_fps >= threshold:
            return k
    raise RuntimeError(
        f"No safe tail-pad found in [{min_pad}, {min_pad + motion_fps}] "
        f"for N={N}, motion_fps={motion_fps}, radar_fps={radar_fps}, "
        f"chirp_per_frame={chirp_per_frame}. Check radar config.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("input_npz",  help="Kimodo-SMPLX-RP AMASS-style NPZ "
                                       "(produced with --model Kimodo-SMPLX-RP --output ...)")
    ap.add_argument("output_npz", help="Path to write RFGen-compatible obj_diff.npz "
                                       "(typically RF-Genesis/output/<name>/obj_diff.npz)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Target FPS (default 30 -- RFGen signal_generator hardcoded rate)")
    ap.add_argument("--gender", default="male", choices=["male", "female", "neutral"],
                    help="Gender tag stored in NPZ. RFGen's pathtracer.py loads "
                         "models/male.ply by default, so 'male' keeps the rest of "
                         "the pipeline working without code changes.")
    ap.add_argument("--no-resample",  action="store_true",
                    help="Skip framerate resampling (use only if Kimodo already exports at 30 fps)")
    ap.add_argument("--no-coord-fix", action="store_true",
                    help="Skip AMASS->y-up coordinate fix")
    ap.add_argument("--face-sensor", action="store_true",
                    help="Additionally rotate 180 deg about y so the body faces the radar "
                         "(RFGen's camera looks toward -z; AMASS forward maps to -z by "
                         "default which leaves the subject facing AWAY from the radar). "
                         "Enable this if you want the subject facing the sensor.")
    ap.add_argument("--tail-pad", default="auto",
                    help="Number of frames to duplicate at the END of the motion to work "
                         "around an off-by-one in RFGen's signal_generator.py:59. "
                         "'auto' (default) computes the smallest pad >= 2 that is "
                         "PROVABLY safe for the given (motion_fps, radar_fps, "
                         "chirp_per_frame) combination -- works for ANY motion length. "
                         "Pass an integer (e.g. --tail-pad 5) to override, or 0 to disable.")
    ap.add_argument("--radar-config", default=None,
                    help="Path to RFGen radar config JSON (e.g. "
                         "RF-Genesis/models/TI1843_config.json). Used by --tail-pad auto "
                         "to read frame_per_second and chirp_per_frame. If omitted, the "
                         "tool searches for models/TI1843_config.json relative to the "
                         "output directory; if not found, falls back to TI1843 defaults "
                         "(frame_per_second=10, chirp_per_frame=128).")
    args = ap.parse_args()

    print(f"[Kimodo->RFGen] Loading {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)

    for k in ("trans", "root_orient", "pose_body"):
        if k not in data.files:
            sys.exit(
                f"ERROR: missing key '{k}' in {args.input_npz}.\n"
                f"  Make sure you exported with the Kimodo-SMPLX-RP model and the\n"
                f"  --output flag, which writes the AMASS-style SMPL-X NPZ. The\n"
                f"  native Kimodo NPZ (posed_joints / global_rot_mats / ...) is\n"
                f"  not supported here -- it is on a different skeleton.")

    trans       = np.asarray(data["trans"]).astype(np.float32)        # (T, 3)
    root_orient = np.asarray(data["root_orient"]).astype(np.float32)  # (T, 3)
    pose_body   = np.asarray(data["pose_body"]).astype(np.float32)    # (T, 63)
    fps_in      = (float(data["mocap_frame_rate"])
                   if "mocap_frame_rate" in data.files else 30.0)

    if pose_body.shape[1] != 63:
        sys.exit(f"ERROR: expected pose_body to be (T, 63), got {pose_body.shape}.")

    print(f"[Kimodo->RFGen] Loaded {trans.shape[0]} frames @ {fps_in:.2f} fps  "
          f"(pose_body {pose_body.shape}, root_orient {root_orient.shape}, "
          f"trans {trans.shape})")

    # ---- (c) FPS resample --------------------------------------------------
    if not args.no_resample:
        before = root_orient.shape[0]
        root_orient, pose_body, trans = resample_motion(
            root_orient, pose_body, trans, fps_in, args.fps)
        print(f"[Kimodo->RFGen] (c) Resampled  {before} -> {root_orient.shape[0]} frames "
              f"({fps_in:.2f} -> {args.fps:.2f} fps)")
    else:
        print(f"[Kimodo->RFGen] (c) Skipped framerate resampling")

    # ---- (b) Coordinate fix ------------------------------------------------
    if not args.no_coord_fix:
        root_orient, trans = amass_to_yup(root_orient, trans)
        print(f"[Kimodo->RFGen] (b) AMASS (z-up) -> y-up applied (-90 deg about x)")

        if args.face_sensor:
            # 180 deg about y in y-up frame: flips forward (-z -> +z).
            R_y180 = np.array([[-1, 0, 0],
                               [ 0, 1, 0],
                               [ 0, 0,-1]], dtype=np.float64)
            R_root = Rotation.from_rotvec(root_orient).as_matrix()
            R_root = R_y180 @ R_root
            root_orient = Rotation.from_matrix(R_root).as_rotvec().astype(np.float32)
            trans = (trans @ R_y180.T).astype(np.float32)
            print(f"[Kimodo->RFGen]     +180 deg about y so subject faces the sensor")
    else:
        print(f"[Kimodo->RFGen] (b) Skipped AMASS->y-up coordinate fix")

    # ---- (a) SMPL-X -> SMPL pose layout ------------------------------------
    pose72 = build_smpl_pose72(root_orient, pose_body)
    print(f"[Kimodo->RFGen] (a) Built SMPL pose-72: {pose72.shape}  "
          f"(L/R hand zero-padded -- SMPL joints 22 & 23)")

    # ---- (d) Tail padding for RFGen off-by-one workaround ------------------
    # RFGen's signal_generator.py:59 does `frames[frame_index + 1]` without
    # bound-checking the (last_frame, total_time) sub-interval. The radar's
    # last chirp burst samples times in this open interval, triggering an
    # IndexError. Compute the smallest safe pad given the radar config so
    # this works for ANY motion length.
    if str(args.tail_pad).lower() == "auto":
        radar_fps, cpf, cfg_path = _read_radar_config(args.radar_config, args.output_npz)
        pad = compute_safe_pad(N=pose72.shape[0],
                               motion_fps=int(args.fps),
                               radar_fps=radar_fps,
                               chirp_per_frame=cpf,
                               min_pad=2)
        cfg_src = cfg_path if cfg_path else "TI1843 defaults (no JSON found)"
        print(f"[Kimodo->RFGen] (d) Auto-pad: radar_fps={radar_fps}, "
              f"chirp_per_frame={cpf} from {cfg_src}")
    else:
        try:
            pad = int(args.tail_pad)
        except ValueError:
            sys.exit(f"ERROR: --tail-pad must be 'auto' or an integer, got "
                     f"{args.tail_pad!r}")
        if pad < 0:
            sys.exit(f"ERROR: --tail-pad must be >= 0, got {pad}")

    if pad > 0:
        pose72 = np.concatenate(
            [pose72, np.tile(pose72[-1:], (pad, 1))], axis=0)
        trans = np.concatenate(
            [trans,  np.tile(trans[-1:],  (pad, 1))], axis=0)
        # Verify against the off-by-one condition (sanity check).
        if str(args.tail_pad).lower() == "auto":
            N_total = pose72.shape[0]
            margin = ((N_total) * radar_fps) % int(args.fps) - (radar_fps - int(args.fps) / cpf)
            print(f"[Kimodo->RFGen] (d) Tail-padded with {pad} static frames "
                  f"-> {N_total} total frames "
                  f"(off-by-one safety margin = {margin:+.3f}, must be >= 0)")
        else:
            print(f"[Kimodo->RFGen] (d) Tail-padded with {pad} static frames "
                  f"-> {pose72.shape[0]} total frames (manual override)")
    else:
        print(f"[Kimodo->RFGen] (d) Tail padding disabled "
              f"(may crash on motion lengths where (N*10) mod 30 < 9.77)")

    # RFGen ignores betas (uses np.zeros(10) in object_diff.py). Save zeros.
    shape = np.zeros(10, dtype=np.float32)

    out_dir = os.path.dirname(os.path.abspath(args.output_npz))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.output_npz,
             pose=pose72,
             shape=shape,
             root_translation=trans.astype(np.float32),
             gender=args.gender)

    print(f"[Kimodo->RFGen] Wrote {args.output_npz}")
    print(f"  pose             : {pose72.shape}  float32")
    print(f"  shape            : {shape.shape}   float32  (zeros -- RFGen ignores betas)")
    print(f"  root_translation : {trans.shape}  float32  meters, y-up")
    print(f"  gender           : {args.gender!r}")
    print()
    print("Done. Now run RFGen:")
    print(f"  cd RF-Genesis")
    print(f"  python run.py -o \"ignored\" -e \"a living room\" "
          f"-n {os.path.basename(os.path.dirname(args.output_npz)) or '<name>'}")


if __name__ == "__main__":
    main()