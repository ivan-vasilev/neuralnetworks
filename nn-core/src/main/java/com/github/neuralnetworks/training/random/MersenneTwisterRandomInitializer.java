package com.github.neuralnetworks.training.random;

import org.uncommons.maths.random.MersenneTwisterRNG;

/**
 * Mersenne twister random initializer
 */
public class MersenneTwisterRandomInitializer extends RandomInitializerImpl {

    public MersenneTwisterRandomInitializer() {
	super(new MersenneTwisterRNG());
    }

    public MersenneTwisterRandomInitializer(float start, float end) {
	super(new MersenneTwisterRNG(), start, end);
    }
}
