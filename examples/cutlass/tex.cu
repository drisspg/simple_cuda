// /* tiled_tex.cu

// Usage:

// nvcc tiled_tex.cu -w \
//   -Xcompiler "-Wfatal-errors" \
//   -lineinfo \
//   -std=c++17 \
//   -I/usr/local/cuda/include \
//   -I/path/to/cutlass/include \
//   -I/path/to/cutlass/tools/util/include

// ./a.out
// */


// #include "cute/tensor.hpp"
// #include "cutlass/numeric_types.h"


// void print_header() {
//   const char* latex_header =
//     "\\documentclass{article}\n"
//     "\\usepackage[a4paper, margin=0.5cm]{geometry}\n"
//     "\\usepackage{adjustbox}\n"
//     "\\usepackage{graphicx}\n"
//     "\\usepackage{lipsum}\n"
//     "\\usepackage{tikz}\n"
//     "\n"
//     "\\begin{document}\n";
//   printf("%s", latex_header);
// }


// void print_footer() {
//   const char* latex_footer = "\\end{document}\n";
//   printf("%s", latex_footer);
// }


// // Copy from mma_atom.hpp
// //
// // Modified to remove printing header and footder, hence allows printing
// // multiple MMAs per TEX file for easier comparisons.
// template <class AtomLayoutMNK,
//           class ValLayoutMNK,
//           class PermutationsMNK,
//           class LayoutC, class ThrIDC,
//           class LayoutA, class ThrIDA,
//           class LayoutB, class ThrIDB>
// void
// print_mma(const char* name,
//           const AtomLayoutMNK& atom_layout_mnk,
//           const ValLayoutMNK& val_layout_mnk,
//           const PermutationsMNK& permutations_mnk,
//           LayoutC const& C, ThrIDC const& TC,    // (m,n) -> (tid,vid)  and  tid -> thr_idx
//           LayoutA const& A, ThrIDA const& TA,    // (m,k) -> (tid,vid)  and  tid -> thr_idx
//           LayoutB const& B, ThrIDB const& TB) {  // (n,k) -> (tid,vid)  and  tid -> thr_idx
//   using namespace cute;

//   printf("\\begin{verbatim}\n");
//   printf("\n%s\n\n", name);

//   printf("  AtomLayoutMNK: "); print(atom_layout_mnk);   printf("\n");
//   printf("   ValLayoutMNK: "); print(val_layout_mnk);    printf("\n");
//   printf("PermutationsMNK: "); print(permutations_mnk); printf("\n\n");

//   printf("LayoutC: "); print(C);  printf("\n");
//   printf(" ThrIDC: "); print(TC); printf("\n");
//   printf("LayoutA: "); print(A);  printf("\n");
//   printf(" ThrIDA: "); print(TA); printf("\n");
//   printf("LayoutB: "); print(B);  printf("\n");
//   printf(" ThrIDB: "); print(TB); printf("\n");
//   printf("\\end{verbatim}\n");

//   printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}%");
//   printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
//          ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");
//   char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
//                               "{rgb,255:red,175;green,255;blue,175}",
//                               "{rgb,255:red,255;green,255;blue,175}",
//                               "{rgb,255:red,255;green,175;blue,175}",
//                               "{rgb,255:red,210;green,210;blue,255}",
//                               "{rgb,255:red,210;green,255;blue,210}",
//                               "{rgb,255:red,255;green,255;blue,210}",
//                               "{rgb,255:red,255;green,210;blue,210}"};

//   // C starting at 0,0
//   for (int m = 0; m < size<0>(C); ++m) {
//     for (int n = 0; n < size<1>(C); ++n) {
//       int thrid   = C(m,n) % size(TC);
//       int val_idx = C(m,n) / size(TC);
//       int thr_idx = TC(thrid);

//       printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
//              color_map[thr_idx % 8],
//              m, n,
//              thr_idx, val_idx);
//     }
//   }

//   // A starting at 0,-size<1>(A)-1
//   for (int m = 0; m < size<0>(A); ++m) {
//     for (int k = 0; k < size<1>(A); ++k) {
//       int thrid   = A(m,k) % size(TA);
//       int val_idx = A(m,k) / size(TA);
//       int thr_idx = TA(thrid);

//       printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
//              color_map[thr_idx % 8],
//              m, k-1-size<1>(A),
//              thr_idx, val_idx);
//     }
//   }

//   // B starting at -size<1>(B)-1,0
//   for (int n = 0; n < size<0>(B); ++n) {
//     for (int k = 0; k < size<1>(B); ++k) {
//       int thrid   = B(n,k) % size(TB);
//       int val_idx = B(n,k) / size(TB);
//       int thr_idx = TB(thrid);

//       printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
//              color_map[thr_idx % 8],
//              k-1-size<1>(B), n,
//              thr_idx, val_idx);
//     }
//   }

//   // A labels
//   for (int m = 0, k = -1; m < size<0>(A); ++m) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), m);
//   }
//   for (int k = 0, m = -1; k < size<1>(A); ++k) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), k);
//   }
//   // B labels
//   for (int n = 0, k = -1; n < size<0>(B); ++n) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, n);
//   }
//   for (int k = 0, n = -1; k < size<1>(B); ++k) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, k);
//   }

//   printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
// }


// template <class TiledCopy,
//           class LayoutS, class ThrIDS,
//           class LayoutD, class ThrIDD>
// void
// print_copy(const char* name,
//            TiledCopy& copy,
//            LayoutS const& S, ThrIDS const& TS,   // (m,n) -> (tid,vid)  and  tid -> thr_idx
//            LayoutD const& D, ThrIDD const& TD) { // (m,n) -> (tid,vid)  and  tid -> thr_idx
//   using namespace cute;

//   CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
//   CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

//   assert(size<0>(S) == size<0>(D));
//   assert(size<1>(S) == size<1>(D));

//   printf("\\begin{verbatim}\n");
//   printf("\n%s\n\n", name);
//   printf("LayoutCopy_TV: "); print(typename TiledCopy::TiledLayout_TV{});  printf("\n");
//   printf(" ShapeTile_MN: "); print(typename TiledCopy::Tiler_MN{});      printf("\n\n");

//   printf("      LayoutS: "); print(S);  printf("\n");
//   printf("       ThrIDS: "); print(TS); printf("\n");
//   printf("      LayoutD: "); print(D);  printf("\n");
//   printf("       ThrIDD: "); print(TD); printf("\n");
//   printf("\\end{verbatim}\n");

//   printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}%");
//   printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
//          ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");
//   char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
//                               "{rgb,255:red,175;green,255;blue,175}",
//                               "{rgb,255:red,255;green,255;blue,175}",
//                               "{rgb,255:red,255;green,175;blue,175}",
//                               "{rgb,255:red,210;green,210;blue,255}",
//                               "{rgb,255:red,210;green,255;blue,210}",
//                               "{rgb,255:red,255;green,255;blue,210}",
//                               "{rgb,255:red,255;green,210;blue,210}"};

//   // S starting at 0,0
//   for (int i = 0; i < size<0>(S); ++i) {
//     for (int j = 0; j < size<1>(S); ++j) {
//       int thrid   = S(i,j) % size(TS);
//       int val_idx = S(i,j) / size(TS);
//       int thr_idx = TS(thrid);

//       printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
//              color_map[thr_idx % 8],
//              i, j,
//              thr_idx, val_idx);
//     }
//   }

//   // D starting at 0,size<1>(S)+3
//   for (int i = 0; i < size<0>(D); ++i) {
//     for (int j = 0; j < size<1>(D); ++j) {
//       int thrid   = D(i,j) % size(TD);
//       int val_idx = D(i,j) / size(TD);
//       int thr_idx = TD(thrid);

//       printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
//              color_map[thr_idx % 8],
//              i + size<0>(S) + 3, j,
//              thr_idx, val_idx);
//     }
//   }

//   // S Labels
//   for (int i = 0, j = -1; i < size<0>(S); ++i) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
//   }
//   for (int j = 0, i = -1; j < size<1>(S); ++j) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
//   }
//   // D Labels
//   for (int i = 0, j = -1; i < size<0>(S); ++i) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, i);
//   }
//   for (int j = 0, i = -1; j < size<1>(D); ++j) {
//     printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, j);
//   }

//   printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
// }

// void print_layouts_for_sm80_cp_async_cachealways() {
//   using namespace cute;

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_16, _1>>,
//       Layout<Shape<_4, _8>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_32, _1>>,
//       Layout<Shape<_4, _8>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_2, _1>>,
//       Layout<Shape<_8, _4>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_16, _1>>,
//       Layout<Shape<_8, _8>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_32, _1>>,
//       Layout<Shape<_8, _8>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_4, _4>>,
//       Layout<Shape<_8, _8>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }

//   {
//     using ST = float;
//     using DT = float;
//     using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
//     auto tiled_copy = TiledCopy<
//       Copy_Atom_Arch,
//       Layout<Shape<_4, _1>>,
//       Layout<Shape<_8, _32>, Stride<_32, _32>>
//     >{};
//     print_copy_content("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
//   }
// }


int main() {
//   print_header();

// //   print_layouts_for_mma();
//   print_layouts_for_sm80_cp_async_cachealways();

//   print_footer();
  return 0;
}
