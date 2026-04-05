from file_writer import FileWriter

def write_fp2_include(f, pp, v):
    f.line("#include \"fp.h\"")
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"fp2.h\"")
    f.line("#include \"utils.h\"")
    f.line()


def write_fp2_file(dir, pp, v):
    path = f"{dir}/fp2.c"
    f = FileWriter(path)

    write_fp2_include(f, pp, v)

