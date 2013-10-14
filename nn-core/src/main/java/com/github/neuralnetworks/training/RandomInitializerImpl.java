package com.github.neuralnetworks.training;

import java.util.Random;

/**
 * 
 * Default implementation of the random initializer using JDK's default Random
 * 
 */
public class RandomInitializerImpl implements RandomInitializer {

    protected Random random;
    protected float start;
    protected float range;

    public RandomInitializerImpl() {
	super();
	this.random = new Random();
	this.start = 0;
	this.range = 1;
    }

    public RandomInitializerImpl(Random random) {
	super();
	this.random = random;
	this.start = 0;
	this.range = 1;
    }

    public RandomInitializerImpl(Random random, float start, float range) {
	super();
	this.random = random;
	this.start = start;
	this.range = range;
    }

    @Override
    public void initialize(float[] array) {
	if (start == 0) {
	    if (range == 1) {
		for (int i = 0; i < array.length; i++) {
		    array[i] = random.nextFloat();
		}
	    } else {
		for (int i = 0; i < array.length; i++) {
		    array[i] = start + random.nextFloat();
		}
	    }
	} else {
	    for (int i = 0; i < array.length; i++) {
		array[i] = start + random.nextFloat() * start;
	    }
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

    public float getRange() {
	return range;
    }

    public void setRange(float range) {
	this.range = range;
    }
}
