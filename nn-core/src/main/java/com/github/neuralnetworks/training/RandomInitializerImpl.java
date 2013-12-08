package com.github.neuralnetworks.training;

import java.util.Random;

/**
 * Default implementation of the random initializer using JDK's default Random
 */
public class RandomInitializerImpl implements RandomInitializer {

    protected Random random;
    protected float start;
    protected float end;

    public RandomInitializerImpl() {
	super();
	this.random = new Random();
	this.start = 0;
	this.end = 1;
    }

    public RandomInitializerImpl(Random random) {
	super();
	this.random = random;
	this.start = 0;
	this.end = 1;
    }

    public RandomInitializerImpl(Random random, float start, float end) {
	super();
	this.random = random;
	this.start = start;
	this.end = end;
    }

    @Override
    public void initialize(float[] array) {
	for (int i = 0; i < array.length; i++) {
	    array[i] = start + random.nextFloat() * (end - start);
	}
    }

    public Random getRandom() {
	return random;
    }

    public void setRandom(Random random) {
	this.random = random;
    }

    public float getStart() {
	return start;
    }

    public void setStart(float start) {
	this.start = start;
    }

    public float getEnd() {
	return end;
    }

    public void setEnd(float end) {
	this.end = end;
    }
}
