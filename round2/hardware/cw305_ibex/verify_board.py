import argparse
import time
from pathlib import Path

REG_CLKSETTINGS = 0x00
REG_CORE_TYPE = 0x02
REG_CORE_REV = 0x03
REG_IDENTIFY = 0x04
REG_CONTROL = 0x05
REG_STATUS = 0x06
REG_FW_ADDR = 0x07
REG_FW_WDATA = 0x08
REG_FW_LOAD = 0x09
REG_RESULT = 0x0A
REG_PC = 0x0B
REG_CYCLE = 0x0C
CTRL_START, CTRL_HOLD_RESET, CTRL_RELEASE_RESET, CTRL_CLEAR = 1, 2, 4, 8


def le32(x):
    return list(int(x).to_bytes(4, "little"))


def rd32(t, r):
    return int.from_bytes(bytearray(t.fpga_read(r, 4)), "little")


def rd8(t, r):
    return int(t.fpga_read(r, 1)[0])


def load_firmware(target, path):
    data = Path(path).read_bytes()
    data += b"\x00" * ((4 - len(data) % 4) % 4)
    target.fpga_write(REG_CONTROL, [CTRL_HOLD_RESET | CTRL_CLEAR])
    for addr in range(0, len(data), 4):
        target.fpga_write(REG_FW_ADDR, le32(addr))
        target.fpga_write(REG_FW_WDATA, list(data[addr : addr + 4]))
        target.fpga_write(REG_FW_LOAD, [0x01])
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--fw", required=True)
    ap.add_argument("--freq", type=float, default=10e6)
    ap.add_argument(
        "--fpga-id",
        default="100t",
        choices=["100t", "35t"],
        help="CW305 FPGA variant (100t or 35t)",
    )
    args = ap.parse_args()

    import chipwhisperer as cw

    scope = cw.scope()
    scope.default_setup()
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

    core = rd8(target, REG_CORE_TYPE)
    rev = rd8(target, REG_CORE_REV)
    ident = rd8(target, REG_IDENTIFY)
    print(
        f"CORE_TYPE=0x{core:02x} (expect 0x49 'I')  REV=0x{rev:02x}  IDENTIFY=0x{ident:02x}"
    )
    ok_id = core == 0x49

    load_firmware(target, args.fw)
    target.fpga_write(REG_CONTROL, [CTRL_RELEASE_RESET])
    time.sleep(0.05)
    target.fpga_write(REG_CONTROL, [CTRL_START])
    time.sleep(0.2)
    status = rd8(target, REG_STATUS)
    cyc = rd32(target, REG_CYCLE)
    pc = rd32(target, REG_PC)
    res = rd32(target, REG_RESULT)
    print(f"STATUS=0x{status:02x}  CYCLE={cyc}  PC=0x{pc:08x}  RESULT=0x{res:08x}")
    ok_run = (590 <= cyc <= 610) and (pc != 0) and (pc != 0xFFFFFFFF)

    scope.dis()
    target.dis()
    if ok_id and ok_run:
        print("IBEX_SILICON_PASS")
    else:
        print(f"IBEX_SILICON_CHECK id_ok={ok_id} run_ok={ok_run}")


if __name__ == "__main__":
    main()
