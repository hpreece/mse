/*
 * mlp_forward_pass.h
 *
 * Standalone C++ implementation of the MLP forward pass for Vynatheya
 * stability classifiers.  Intended to exactly reproduce the scikit-learn
 * MLPClassifier predict_proba output using the exported weight arrays
 * from stability_mlp_weights.h.
 *
 * Architecture (all three models):
 *   4 hidden layers of 50 neurons each, sigmoid activation throughout.
 *   Output layer: single neuron, sigmoid -> P(unstable).
 *
 * Weight layout (sklearn convention, stored row-major):
 *   W has shape (n_in, n_out), flattened as W[i * n_out + j].
 *   Forward: out[j] = sigmoid( sum_i( W[i*n_out + j] * in[i] ) + b[j] )
 */

#ifndef MLP_FORWARD_PASS_H
#define MLP_FORWARD_PASS_H

#include <cmath>
#include "stability_mlp_weights.h"

/* ------------------------------------------------------------------ */
/* Sigmoid                                                             */
/* ------------------------------------------------------------------ */

static inline double mlp_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/* ------------------------------------------------------------------ */
/* Generic single-layer forward pass                                   */
/*                                                                     */
/* Computes:  out[j] = sigmoid( sum_i(W[i*n_out+j] * in[i]) + b[j] )  */
/* for j = 0 .. n_out-1                                                */
/* ------------------------------------------------------------------ */

static inline void mlp_layer_forward(const double *input, int n_in,
                                     const double *W, const double *b,
                                     double *output, int n_out) {
    for (int j = 0; j < n_out; j++) {
        double sum = b[j];
        for (int i = 0; i < n_in; i++) {
            sum += W[i * n_out + j] * input[i];
        }
        output[j] = mlp_sigmoid(sum);
    }
}

/* ------------------------------------------------------------------ */
/* Triple model: 6 features -> P(unstable)                             */
/*                                                                     */
/* Features: qi, qo, a_in/a_out, ei, eo, i_mut/pi                     */
/* 5 layers: L0(6->50), L1(50->50), L2(50->50), L3(50->50), L4(50->1) */
/* ------------------------------------------------------------------ */

static double mlp_predict_triple(const double features[6]) {
    double h0[MLP_TRIPLE_L0_N_OUT];
    double h1[MLP_TRIPLE_L1_N_OUT];
    double h2[MLP_TRIPLE_L2_N_OUT];
    double h3[MLP_TRIPLE_L3_N_OUT];
    double out[MLP_TRIPLE_L4_N_OUT];

    mlp_layer_forward(features, MLP_TRIPLE_L0_N_IN,
                      mlp_triple_W0, mlp_triple_b0,
                      h0, MLP_TRIPLE_L0_N_OUT);

    mlp_layer_forward(h0, MLP_TRIPLE_L1_N_IN,
                      mlp_triple_W1, mlp_triple_b1,
                      h1, MLP_TRIPLE_L1_N_OUT);

    mlp_layer_forward(h1, MLP_TRIPLE_L2_N_IN,
                      mlp_triple_W2, mlp_triple_b2,
                      h2, MLP_TRIPLE_L2_N_OUT);

    mlp_layer_forward(h2, MLP_TRIPLE_L3_N_IN,
                      mlp_triple_W3, mlp_triple_b3,
                      h3, MLP_TRIPLE_L3_N_OUT);

    mlp_layer_forward(h3, MLP_TRIPLE_L4_N_IN,
                      mlp_triple_W4, mlp_triple_b4,
                      out, MLP_TRIPLE_L4_N_OUT);

    return out[0];
}

/* ------------------------------------------------------------------ */
/* 2+2 Quad model: 11 features -> P(unstable)                          */
/*                                                                     */
/* Features: qi1, qi2, qo, ali1o, ali2o, ei1, ei2, eo,                 */
/*           ii1i2/pi, ii1o/pi, ii2o/pi                                */
/* 5 layers: L0(11->50), L1-L3(50->50), L4(50->1)                     */
/* ------------------------------------------------------------------ */

static double mlp_predict_quad_2p2(const double features[11]) {
    double h0[MLP_QUAD_2P2_L0_N_OUT];
    double h1[MLP_QUAD_2P2_L1_N_OUT];
    double h2[MLP_QUAD_2P2_L2_N_OUT];
    double h3[MLP_QUAD_2P2_L3_N_OUT];
    double out[MLP_QUAD_2P2_L4_N_OUT];

    mlp_layer_forward(features, MLP_QUAD_2P2_L0_N_IN,
                      mlp_quad_2p2_W0, mlp_quad_2p2_b0,
                      h0, MLP_QUAD_2P2_L0_N_OUT);

    mlp_layer_forward(h0, MLP_QUAD_2P2_L1_N_IN,
                      mlp_quad_2p2_W1, mlp_quad_2p2_b1,
                      h1, MLP_QUAD_2P2_L1_N_OUT);

    mlp_layer_forward(h1, MLP_QUAD_2P2_L2_N_IN,
                      mlp_quad_2p2_W2, mlp_quad_2p2_b2,
                      h2, MLP_QUAD_2P2_L2_N_OUT);

    mlp_layer_forward(h2, MLP_QUAD_2P2_L3_N_IN,
                      mlp_quad_2p2_W3, mlp_quad_2p2_b3,
                      h3, MLP_QUAD_2P2_L3_N_OUT);

    mlp_layer_forward(h3, MLP_QUAD_2P2_L4_N_IN,
                      mlp_quad_2p2_W4, mlp_quad_2p2_b4,
                      out, MLP_QUAD_2P2_L4_N_OUT);

    return out[0];
}

/* ------------------------------------------------------------------ */
/* 3+1 Quad model: 11 features -> P(unstable)                          */
/*                                                                     */
/* Features: qi, qm, qo, alim, almo, ei, em, eo,                      */
/*           iim/pi, iio/pi, imo/pi                                    */
/* 5 layers: L0(11->50), L1-L3(50->50), L4(50->1)                     */
/* ------------------------------------------------------------------ */

static double mlp_predict_quad_3p1(const double features[11]) {
    double h0[MLP_QUAD_3P1_L0_N_OUT];
    double h1[MLP_QUAD_3P1_L1_N_OUT];
    double h2[MLP_QUAD_3P1_L2_N_OUT];
    double h3[MLP_QUAD_3P1_L3_N_OUT];
    double out[MLP_QUAD_3P1_L4_N_OUT];

    mlp_layer_forward(features, MLP_QUAD_3P1_L0_N_IN,
                      mlp_quad_3p1_W0, mlp_quad_3p1_b0,
                      h0, MLP_QUAD_3P1_L0_N_OUT);

    mlp_layer_forward(h0, MLP_QUAD_3P1_L1_N_IN,
                      mlp_quad_3p1_W1, mlp_quad_3p1_b1,
                      h1, MLP_QUAD_3P1_L1_N_OUT);

    mlp_layer_forward(h1, MLP_QUAD_3P1_L2_N_IN,
                      mlp_quad_3p1_W2, mlp_quad_3p1_b2,
                      h2, MLP_QUAD_3P1_L2_N_OUT);

    mlp_layer_forward(h2, MLP_QUAD_3P1_L3_N_IN,
                      mlp_quad_3p1_W3, mlp_quad_3p1_b3,
                      h3, MLP_QUAD_3P1_L3_N_OUT);

    mlp_layer_forward(h3, MLP_QUAD_3P1_L4_N_IN,
                      mlp_quad_3p1_W4, mlp_quad_3p1_b4,
                      out, MLP_QUAD_3P1_L4_N_OUT);

    return out[0];
}

#endif /* MLP_FORWARD_PASS_H */
