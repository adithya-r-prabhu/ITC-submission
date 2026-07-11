import argparse
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

REG_CLKSETTINGS = 0x00
REG_CONTROL = 0x05
REG_STATUS = 0x06
REG_FW_ADDR = 0x07
REG_FW_WDATA = 0x08
REG_FW_LOAD = 0x09
REG_RESULT = 0x0A
REG_PC = 0x0B
REG_FAULT = 0x0E
CTRL_START, CTRL_HOLD_RESET, CTRL_RELEASE_RESET, CTRL_CLEAR = 1, 2, 4, 8

TARGET_INSTR, TARGET_DATA = 1, 2


def le32(x):
    return list(int(x).to_bytes(4, "little"))


def rd32(target, reg):
    return int.from_bytes(bytearray(target.fpga_read(reg, 4)), "little")


def fault_word(enable, target, bit, pol):
    return (
        int(bool(enable))
        | ((target & 0x3) << 1)
        | ((pol & 0x1) << 4)
        | ((bit & 0x1F) << 8)
    )


def load_firmware(target, path):
    data = Path(path).read_bytes()
    data += b"\x00" * ((4 - len(data) % 4) % 4)
    target.fpga_write(REG_CONTROL, [CTRL_HOLD_RESET | CTRL_CLEAR])
    for addr in range(0, len(data), 4):
        target.fpga_write(REG_FW_ADDR, le32(addr))
        target.fpga_write(REG_FW_WDATA, list(data[addr : addr + 4]))
        target.fpga_write(REG_FW_LOAD, [0x01])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--fw", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--core", default="ibex")
    ap.add_argument("--n-clean", type=int, default=50, help="golden (fault-off) traces")
    ap.add_argument("--reps", type=int, default=1, help="captures per fault site")
    ap.add_argument("--targets", default="instr,data", help="comma list of instr,data")
    ap.add_argument("--bits", default="0-31", help="bit range, e.g. 0-31 or 0,2,7")
    ap.add_argument("--pols", default="0,1", help="stuck-at polarities to sweep")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--freq", type=float, default=10e6)
    ap.add_argument(
        "--fpga-id",
        default="100t",
        choices=["100t", "35t"],
        help="CW305 FPGA variant (100t or 35t)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tgt_map = {"instr": TARGET_INSTR, "data": TARGET_DATA}
    targets = [tgt_map[t.strip()] for t in args.targets.split(",") if t.strip()]
    if "-" in args.bits:
        lo, hi = args.bits.split("-")
        bit_list = list(range(int(lo), int(hi) + 1))
    else:
        bit_list = [int(b) for b in args.bits.split(",") if b.strip() != ""]
    pol_list = [int(p) for p in args.pols.split(",") if p.strip() != ""]

    import chipwhisperer as cw

    scope = cw.scope()
    scope.default_setup()
    scope.adc.samples = args.samples
    scope.trigger.triggers = "tio4"

    target = cw.target(
        scope,
        cw.targets.CW305,
        bsfile=args.bitstream,
        fpga_id=args.fpga_id,
        force=True,
        slurp=False,
    )
    target.vccint_set(1.0)
    target.pll.pll_enable_set(True)
    target.pll.pll_outfreq_set(args.freq, 1)
    target.pll.pll_outenable_set(False, 0)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outenable_set(False, 2)
    target.fpga_write(REG_CLKSETTINGS, [0x09])
    scope.clock.adc_src = "extclk_x4"
    scope.clock.reset_adc()
    time.sleep(0.2)

    load_firmware(target, args.fw)
    target.fpga_write(REG_CONTROL, [CTRL_RELEASE_RESET])

    def run_once():
        scope.arm()
        target.fpga_write(REG_CONTROL, [CTRL_START])
        to = scope.capture()
        return (
            scope.get_last_trace(),
            rd32(target, REG_RESULT),
            target.fpga_read(REG_STATUS, 1)[0],
            to,
        )

    target.fpga_write(REG_FAULT, le32(0))
    clean_traces, clean_results = [], []
    for i in range(args.n_clean):
        tr, res, _st, to = run_once()
        if to:
            continue
        clean_traces.append(tr)
        clean_results.append(res)
    golden = Counter(clean_results).most_common(1)[0][0] if clean_results else 0
    print(f"  golden result = 0x{golden:08x} over {len(clean_traces)} clean runs")

    f_traces, f_tgt, f_bit, f_pol, f_obs, f_trap = [], [], [], [], [], []
    for tg in targets:
        for b in bit_list:
            for p in pol_list:
                fw = fault_word(1, tg, b, p)
                for _r in range(args.reps):
                    target.fpga_write(REG_FAULT, le32(fw))
                    tr, res, status, to = run_once()
                    if to:
                        continue
                    f_traces.append(tr)
                    f_tgt.append(tg)
                    f_bit.append(b)
                    f_pol.append(p)
                    f_obs.append(int((res & 0xFFFFFFFF) != golden))
                    f_trap.append(int((status >> 2) & 1))
    target.fpga_write(REG_FAULT, le32(0))

    n = len(f_obs)
    n_obs = int(np.sum(f_obs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        clean_traces=np.array(clean_traces, dtype=np.float32),
        fault_traces=np.array(f_traces, dtype=np.float32),
        target=np.array(f_tgt, dtype=np.int8),
        bit=np.array(f_bit, dtype=np.int8),
        pol=np.array(f_pol, dtype=np.int8),
        observable=np.array(f_obs, dtype=np.int8),
        trap=np.array(f_trap, dtype=np.int8),
        golden_result=np.uint32(golden),
        workload=args.workload,
        core=args.core,
        samples=args.samples,
        freq=args.freq,
        bitstream_sha256=sha256(Path(args.bitstream).read_bytes()).hexdigest(),
        firmware_sha256=sha256(Path(args.fw).read_bytes()).hexdigest(),
    )
    print(
        f"  saved {out}: {n} stuck-at trials, {n_obs} observable "
        f"({100 * n_obs / max(1, n):.1f}%), {len(clean_traces)} clean"
    )
    scope.dis()
    target.dis()


if __name__ == "__main__":
    main()
