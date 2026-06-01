"""Check 7 from training_prompt_full_run.md: assert span masking for audio
produces CONTIGUOUS spans aligned to codebook boundaries (stride=2).

Run from the nano4M repo root:
    PYTHONPATH=. python scripts/check_span_mask.py
"""
import sys
import torch

from nanofm.data.multimodal.masking import _span_positions, SimpleMultimodalMasking


def main():
    fails = 0
    print("== _span_positions sanity (10 trials @ stride=2) ==")
    for trial in range(10):
        in_pos, tg_pos = _span_positions(num_tokens=512, n_input=120,
                                          n_target=80, stride=2)
        in_contig = (in_pos[-1] - in_pos[0]) == (len(in_pos) - 1)
        tg_contig = (tg_pos[-1] - tg_pos[0]) == (len(tg_pos) - 1)
        cb_in = int(in_pos[0]) % 2 == 0 and (int(in_pos[-1]) + 1) % 2 == 0
        cb_tg = int(tg_pos[0]) % 2 == 0 and (int(tg_pos[-1]) + 1) % 2 == 0
        disjoint = set(in_pos.tolist()).isdisjoint(set(tg_pos.tolist()))
        print(f" trial {trial}: in[{int(in_pos[0])},{int(in_pos[-1])}] tg[{int(tg_pos[0])},{int(tg_pos[-1])}]"
              f" contig=({in_contig},{tg_contig}) cb=({cb_in},{cb_tg}) disjoint={disjoint}")
        if not (in_contig and tg_contig and cb_in and cb_tg and disjoint):
            fails += 1

    print("\n== SimpleMultimodalMasking end-to-end ==")
    masker = SimpleMultimodalMasking(
        modalities=["tok_rgb@196", "tok_audio@512"],
        vocab_sizes=[16384, 2048],
        max_seq_lens=[196, 512],
        input_alphas=[1.0, 1.0],
        target_alphas=[1.0, 1.0],
        input_tokens_range=(64, 128),
        target_tokens_range=(64, 128),
        overlap_vocab=False,
        span_modalities={"tok_audio@512": 2},
    )
    data = {
        "tok_rgb@196":   torch.randint(0, 16384, (196,)),
        "tok_audio@512": torch.randint(0, 2048,  (512,)),
    }
    out = masker(data)
    # Check that audio positions in enc/dec are contiguous and cb-aligned
    audio_mod_idx = 1
    enc_audio_mask = out["enc_modalities"] == audio_mod_idx
    dec_audio_mask = out["dec_modalities"] == audio_mod_idx
    enc_pad = out["enc_pad_mask"]
    dec_pad = out["dec_pad_mask"]
    enc_a = out["enc_positions"][enc_audio_mask & enc_pad]
    dec_a = out["dec_positions"][dec_audio_mask & dec_pad]
    print(f"  enc audio positions: count={len(enc_a)}"
          + (f" range=[{int(enc_a.min())},{int(enc_a.max())}]" if len(enc_a) else ""))
    print(f"  dec audio positions: count={len(dec_a)}"
          + (f" range=[{int(dec_a.min())},{int(dec_a.max())}]" if len(dec_a) else ""))
    if len(enc_a) > 0:
        contig = (enc_a.max() - enc_a.min()) == (len(enc_a) - 1)
        cb     = int(enc_a.min()) % 2 == 0 and (int(enc_a.max()) + 1) % 2 == 0
        print(f"  enc contig={contig} cb_aligned={cb}")
        if not (contig and cb): fails += 1
    if len(dec_a) > 0:
        contig = (dec_a.max() - dec_a.min()) == (len(dec_a) - 1)
        cb     = int(dec_a.min()) % 2 == 0 and (int(dec_a.max()) + 1) % 2 == 0
        print(f"  dec contig={contig} cb_aligned={cb}")
        if not (contig and cb): fails += 1
    # Check RGB still gets random (non-contiguous most of the time)
    rgb_pos = out["enc_positions"][(out["enc_modalities"] == 0) & enc_pad]
    if len(rgb_pos) > 4:
        rgb_contig = (rgb_pos.max() - rgb_pos.min()) == (len(rgb_pos) - 1)
        print(f"  rgb enc positions n={len(rgb_pos)} contig={rgb_contig} "
              f"(expected False most of the time → random)")

    print(f"\nFAILURES: {fails}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
