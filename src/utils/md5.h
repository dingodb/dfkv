/* Compact MD5 (RFC 1321), public-domain style, header-only.
 * Used for deterministic key->id derivation and Ketama ring points so the KV
 * client routing matches dingofs's MD5-based KetamaConHash family. */
#ifndef DFKV_MD5_H_
#define DFKV_MD5_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace dfkv {
namespace md5detail {

struct Ctx {
  uint32_t a, b, c, d;
  uint64_t len;
  uint8_t buf[64];
  size_t buflen;
};

inline uint32_t rol(uint32_t x, int s) { return (x << s) | (x >> (32 - s)); }

inline void Block(Ctx& m, const uint8_t* p) {
  static const uint32_t K[64] = {
      0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
      0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
      0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
      0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
      0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
      0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
      0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
      0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391};
  static const int S[64] = {7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                            5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                            4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                            6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
  uint32_t M[16];
  for (int i = 0; i < 16; ++i)
    M[i] = uint32_t(p[i*4]) | (uint32_t(p[i*4+1])<<8) | (uint32_t(p[i*4+2])<<16) | (uint32_t(p[i*4+3])<<24);
  uint32_t A=m.a,B=m.b,C=m.c,D=m.d;
  for (int i = 0; i < 64; ++i) {
    uint32_t F; int g;
    if (i < 16)      { F=(B&C)|(~B&D); g=i; }
    else if (i < 32) { F=(D&B)|(~D&C); g=(5*i+1)&15; }
    else if (i < 48) { F=B^C^D;        g=(3*i+5)&15; }
    else             { F=C^(B|~D);     g=(7*i)&15; }
    F = F + A + K[i] + M[g];
    A=D; D=C; C=B; B=B+rol(F,S[i]);
  }
  m.a+=A; m.b+=B; m.c+=C; m.d+=D;
}

inline void Init(Ctx& m){ m.a=0x67452301;m.b=0xefcdab89;m.c=0x98badcfe;m.d=0x10325476;m.len=0;m.buflen=0; }
inline void Update(Ctx& m, const uint8_t* data, size_t n){
  m.len += n;
  while (n) {
    size_t take = 64 - m.buflen; if (take > n) take = n;
    std::memcpy(m.buf + m.buflen, data, take);
    m.buflen += take; data += take; n -= take;
    if (m.buflen == 64) { Block(m, m.buf); m.buflen = 0; }
  }
}
inline void Final(Ctx& m, uint8_t out[16]){
  uint64_t bits = m.len * 8;
  uint8_t pad = 0x80; Update(m, &pad, 1);
  uint8_t zero = 0; while (m.buflen != 56) Update(m, &zero, 1);
  uint8_t lenb[8]; for (int i=0;i<8;++i) lenb[i]=uint8_t(bits>>(8*i)); Update(m, lenb, 8);
  uint32_t v[4]={m.a,m.b,m.c,m.d};
  for (int i=0;i<4;++i){ out[i*4]=v[i]&0xff; out[i*4+1]=(v[i]>>8)&0xff; out[i*4+2]=(v[i]>>16)&0xff; out[i*4+3]=(v[i]>>24)&0xff; }
}

}  // namespace md5detail

inline void Md5(const void* data, size_t n, uint8_t out[16]) {
  md5detail::Ctx m; md5detail::Init(m);
  md5detail::Update(m, static_cast<const uint8_t*>(data), n);
  md5detail::Final(m, out);
}

// first 8 bytes of the digest, little-endian -> uint64
inline uint64_t Md5_64(const std::string& s) {
  uint8_t d[16]; Md5(s.data(), s.size(), d);
  uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(d[i]) << (8 * i);
  return v;
}
// first 4 bytes little-endian -> uint32 (Ketama point hash)
inline uint32_t Md5_32(const std::string& s) {
  uint8_t d[16]; Md5(s.data(), s.size(), d);
  return uint32_t(d[0]) | (uint32_t(d[1])<<8) | (uint32_t(d[2])<<16) | (uint32_t(d[3])<<24);
}

}  // namespace dfkv

#endif  // DFKV_MD5_H_
