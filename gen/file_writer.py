class FileWriter:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(file_name, "w")
        self.indent_lvl = 0

    def indent(self):
        self.indent_lvl += 1

    def unindent(self):
        if self.indent_lvl > 0:
            self.indent_lvl -= 1

    def line(self, str=""):
        tab = "    "
        self.file.write(f"{tab*self.indent_lvl}{str}\n")

    def comment(self, str):
        self.line("// " + str)

    def banner(self, str):
        banner_len = 100
        padding_len = (banner_len - len(str))//2
        self.comment("=" * banner_len)
        self.comment(" " * padding_len + str + " " * padding_len)
        self.comment("=" * banner_len)
        self.line()

    # start a new block
    def start(self, str):
        self.line(str)
        self.line("{")
        self.indent()

    # end the current block
    def end(self):
        self.unindent()
        self.line("}")
        self.line()
        