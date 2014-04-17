package com.github.neuralnetworks.training.random;

import com.amd.aparapi.Kernel;

/**
 * Pseudo Random Number Generator Kernel
 * 
 * with basic methods for different random ranges
 * 
 * @author Marcin Kotz (https://github.com/er453r/Aparapi-RNG)
 */
public abstract class PRNGKernel extends Kernel {
    /** Integer size used for seed sizes */
    public final static int INTEGER_SIZE = 4;

    /**
     * Each PRNG based on this class should implement this as returning int from
     * range: <Integer.MIN_VALUE; Integer.MAX_VALUE>
     * 
     * @return pseudo random int from range: &lt;Integer.MIN_VALUE;
     *         Integer.MAX_VALUE>
     */
    public abstract int random();

    /**
     * Float version of random
     * 
     * @return pseudo random float from range: &lt;0.0; 1.0>
     */
    public float random01() {
	float value = random();

	if (value < 0)
	    return value / Integer.MIN_VALUE;
	else if (value > 0)
	    return value / Integer.MAX_VALUE;
	else
	    return value;
    }

    /**
     * Float version of random
     * 
     * @return pseudo random float from range: &lt;-1.0; 1.0>
     */
    public float randomn11() {
	float value = random();

	if (value < 0)
	    return -value / Integer.MIN_VALUE;
	else if (value > 0)
	    return value / Integer.MAX_VALUE;
	else
	    return value;
    }

    /**
     * Returns the next pseudorandom, Gaussian ("normally") distributed double
     * value with mean 0.0 and standard deviation 1.0 from this random number
     * generator's sequence.
     * 
     * This uses the polar method of G. E. P. Box, M. E. Muller, and G.
     * Marsaglia, as described by Donald E. Knuth in The Art of Computer
     * Programming, Volume 2: Seminumerical Algorithms, section 3.4.1,
     * subsection C, algorithm P.
     * 
     * @return the next pseudorandom, Gaussian ("normally") distributed double
     *         value with mean 0.0 and standard deviation 1.0 from this random
     *         number generator's sequence.
     */
    public float randomGaussian() {
	float v1 = 1, v2, s = 1;
	boolean found = false;

	while (!found) {
	    v1 = randomn11();
	    v2 = randomn11();
	    s = v1 * v1 + v2 * v2;

	    found = s < 1 && s > 0;
	}

	return v1 * sqrt(-2 * log(s) / s);
    }
}
