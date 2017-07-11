package com.github.neuralnetworks.training.random;

import org.uncommons.maths.binary.BinaryUtils;
import org.uncommons.maths.random.DefaultSeedGenerator;

import com.aparapi.Range;

/**
 * PRNG implementing XORShiftRNG from Uncommons Maths library (
 * http://maths.uncommons.org/ )
 * 
 * A Java implementation of the very fast PRNG described by George Marsaglia. It
 * has a period of about 2^160, which although much shorter than the Mersenne
 * Twister's, is still significantly longer than that of java.util.Random. This
 * is the RNG to use when performance is the primary concern. It can be up to
 * twice as fast as the Mersenne Twister.
 * 
 * Requires 20 (5 * 4) bytes of memory per thread
 * 
 * @author Marcin Kotz (https://github.com/er453r/Aparapi-RNG)
 */
public abstract class XORShiftKernel extends PRNGKernel {
    /** XORShift has 5 states, so it needs 5 seeds */
    public final static int SEED_SIZE = 5;

    /** States for each thread of the XORShift kernel */
    final int states[];

    /**
     * Initializes the XORShift PRNG
     * 
     * @param maximumRange
     *            maximum range that will be used with this kernel (sets seed
     *            values)
     * @param seeds
     *            set seeds explicitly
     */
    public XORShiftKernel(Range maximumRange, int[] seeds) {
	if (maximumRange.getDims() != 1)
	    throw new IllegalArgumentException("Only 1-dimensional ranges supported!");

	int maxThreads = maximumRange.getGlobalSize(0);

	states = new int[SEED_SIZE * maxThreads];

	if (seeds == null)
	    seeds = BinaryUtils.convertBytesToInts(DefaultSeedGenerator.getInstance().generateSeed(SEED_SIZE * maxThreads * INTEGER_SIZE));

	if (SEED_SIZE * maxThreads != seeds.length)
	    throw new IllegalArgumentException(String.format("Wrong size of seeds for threads! Expected %d, got %d, for %d threads.", SEED_SIZE * maxThreads, seeds.length, maxThreads));

	for (int n = 0; n < states.length; n++)
	    states[n] = seeds[n];
    }

    /**
     * Shortcut to XORShiftKernel(Range maximumRange, int[] seeds)
     * 
     * @param maximumRange
     *            maximum range size
     */
    public XORShiftKernel(int maximumRange) {
	this(Range.create(maximumRange), null);
    }

    /** Implements PRNGKernel method */
    @Override
    public int random() {
	int offset = SEED_SIZE * getGlobalId();

	int t = (states[offset + 0] ^ (states[offset + 0] >> 7));

	states[offset + 0] = states[offset + 1];
	states[offset + 1] = states[offset + 2];
	states[offset + 2] = states[offset + 3];
	states[offset + 3] = states[offset + 4];
	states[offset + 4] = (states[offset + 4] ^ (states[offset + 4] << 6)) ^ (t ^ (t << 13));

	return (states[offset + 1] + states[offset + 1] + 1) * states[offset + 4];
    }
}