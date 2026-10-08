// C testbench: sv_engine on one PC slice. usage: tb ins.bin state.bin out.bin
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <ap_int.h>
typedef ap_uint<256> w256;
#ifdef SV_WIDE
typedef ap_uint<512> pw_t;
#else
typedef w256 pw_t;
#endif
extern "C" void sv_engine(pw_t *st, long ins_off);
static std::vector<char> slurp(const char *f) {
    FILE *fp = fopen(f, "rb"); if (!fp) { perror(f); exit(1); }
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    std::vector<char> b(n); if (fread(b.data(), 1, n, fp) != (size_t)n) exit(1); fclose(fp); return b;
}
int main(int argc, char **argv) {
    auto ins = slurp(argv[1]); auto st = slurp(argv[2]);
    size_t nw = st.size() / 32;
    while (ins.size() % 32) ins.push_back(0);
    size_t ni = ins.size() / 32;
    std::vector<w256> s(nw + ni);
    for (size_t i = 0; i < nw; i++)
        for (int b = 0; b < 32; b++) s[i].range(8 * b + 7, 8 * b) = (unsigned char)st[32 * i + b];
    for (size_t i = 0; i < ni; i++)
        for (int b = 0; b < 32; b++) s[nw + i].range(8 * b + 7, 8 * b) = (unsigned char)ins[32 * i + b];
#ifdef SV_WIDE
    if ((nw + ni) % 2) { s.push_back(0); }
    std::vector<pw_t> sw(s.size() / 2);
    for (size_t i = 0; i < sw.size(); i++) { sw[i].range(255, 0) = s[2 * i]; sw[i].range(511, 256) = s[2 * i + 1]; }
    sv_engine(sw.data(), (long)nw);
    for (size_t i = 0; i < sw.size(); i++) { s[2 * i] = sw[i].range(255, 0); s[2 * i + 1] = sw[i].range(511, 256); }
#else
    sv_engine(s.data(), (long)nw);
#endif
    for (size_t i = 0; i < nw; i++)
        for (int b = 0; b < 32; b++) st[32 * i + b] = (char)(unsigned)s[i].range(8 * b + 7, 8 * b);
    FILE *fp = fopen(argv[3], "wb"); fwrite(st.data(), 1, st.size(), fp); fclose(fp);
    return 0;
}
