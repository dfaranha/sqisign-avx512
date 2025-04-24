import cProfile
import pstats

from sage.all import proof

from sqisign.sqisign import SQIsign
from test.test_verification import parse_kat_file

NUM_KATS = 10


def bench_verification(level=1):
    sqisign = SQIsign(level)
    filename = f"assets/PQCsignKAT_{sqisign.sk_len}_SQIsign_lvl{level}.rsp"

    data = []

    for kat_data in parse_kat_file(filename)[:NUM_KATS]:
        # Extract out signture from sm
        message = kat_data["msg"]
        signature = kat_data["sm"][: -kat_data["mlen"]]
        public_key = kat_data["pk"]
        data.append([message, signature, public_key])

    # Run SQIsign verification for benchmarking

    # Start profiler
    pr_verify = cProfile.Profile()
    pr_verify.enable()

    # Run verification
    for message, signature, public_key in data:
        assert sqisign.verify(message, signature, public_key)

    # End profiler
    pr_verify.disable()
    pr_verify.dump_stats("./bench/verify.cProfile")
    p = pstats.Stats("./bench/verify.cProfile")
    p.strip_dirs().sort_stats("cumtime").print_stats(25)


if __name__ == "__main__":
    proof.all(False)
    bench_verification(5)
