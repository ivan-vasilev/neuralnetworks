package com.github.neuralnetworks.training;

import java.util.Random;

/**
 *
 * Default implementation of the random initializer using JDK's default Random
 *
 */
public class RandomInitializerImpl implements RandomInitializer {

	protected Random random;
	protected float max;

	public RandomInitializerImpl() {
		super();
		this.random = new Random();
		this.max = 1;
	}

	public RandomInitializerImpl(Random random) {
		super();
		this.random = random;
		this.max = 1;
	}

	public RandomInitializerImpl(Random random, float max) {
		super();
		this.random = random;
		this.max = max;
	}

	@Override
	public void initialize(float[] array) {
		if (max == 1) {
			for (int i = 0; i < array.length; i++) {
				array[i] = random.nextFloat();
			}
		} else {
			for (int i = 0; i < array.length; i++) {
				array[i] = random.nextFloat() * max;
			}
		}
	}

	public Random getRandom() {
		return random;
	}

	public void setRandom(Random random) {
		this.random = random;
	}

	public float getMax() {
		return max;
	}

	public void setMax(float max) {
		this.max = max;
	}
}
