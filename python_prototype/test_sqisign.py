from sqisign import SQIsign


def read_kat_int(kat_line):
    return int(kat_line.split(" = ")[-1])


def read_kat_hexbytes(kat_line):
    return bytes.fromhex(kat_line.split(" = ")[-1])


def parse_kat_block(b):
    return {
        "count": read_kat_int(b[0]),
        "seed": read_kat_hexbytes(b[1]),
        "mlen": read_kat_int(b[2]),
        "msg": read_kat_hexbytes(b[3]),
        "pk": read_kat_hexbytes(b[4]),
        "sk": read_kat_hexbytes(b[5]),
        "smlen": read_kat_int(b[6]),
        "sm": read_kat_hexbytes(b[7]),
    }


def parse_kat_file(filename):
    with open(filename, "r") as f:
        kat_lines = f.readlines()[2:]
    return [parse_kat_block(kat_lines[i : i + 9]) for i in range(0, len(kat_lines), 9)]


def generic_KAT_test(level=1, num_tests=100):
    sqisign = SQIsign(level)
    kat_tests = parse_kat_file(
        f"../assets/PQCsignKAT_{sqisign.sk_len}_SQIsign_lvl{level}.rsp"
    )

    print(f"Testing {num_tests} of total {len(kat_tests)} tests for level {level}")
    for i, test in enumerate(kat_tests[:num_tests]):
        print(".", end="", flush=True)

        # Assert lengths match expectation from KAT
        assert len(test["msg"]) == test["mlen"]
        assert len(test["sm"]) == test["smlen"]

        # Ensure the message matches what is sent with sm
        assert test["sm"][-test["mlen"] :] == test["msg"]

        # Extract out signture from sm
        signature = test["sm"][: -test["mlen"]]

        # Run SQIsign verification
        assert sqisign.verify(test["msg"], signature, test["pk"])
    print(f" level {level} tests passed!")


def test_level_1():
    generic_KAT_test(1, 5)


def test_level_3():
    generic_KAT_test(3, 5)


def test_level_5():
    generic_KAT_test(5, 5)


if __name__ == "__main__":
    # test_level_1()
    test_level_3()
    # test_level_5()
