package com.github.neuralnetworks.training.random;

import java.util.Random;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * @author tmey
 */
public class NormalDistributionInitializer implements RandomInitializer
{

	private static final long serialVersionUID = 1L;

	private final NormalDistribution normalDistribution;
	private Long seed = null;

	public NormalDistributionInitializer()
	{
		this.seed = new Random().nextLong();
		normalDistribution = new NormalDistribution(0, 0.0001);
		normalDistribution.reseedRandomGenerator(seed);
	}

	public NormalDistributionInitializer(double mean, double standardDeviation)
	{
		this.seed = new Random().nextLong();
		normalDistribution = new NormalDistribution(mean, standardDeviation);
		normalDistribution.reseedRandomGenerator(seed);

	}


	public NormalDistributionInitializer(RandomGenerator rng, double mean, double standardDeviation)
	{
		normalDistribution = new NormalDistribution(rng, mean, standardDeviation, 1e-9);
	}

	public NormalDistributionInitializer(RandomGenerator rng, double mean, double standardDeviation, long seed)
	{
		this.seed = seed;
		normalDistribution = new NormalDistribution(rng, mean, standardDeviation, 1e-9);
	}

	@Override
    public void initialize(Tensor t) {
        float[] elements = t.getElements();
        t.forEach(i -> elements[i] = (float)normalDistribution.sample());
    }

	public double getMean()
	{
		return normalDistribution.getMean();
	}

	public double getStandardDeviation()
	{
		return normalDistribution.getStandardDeviation();
	}

	public boolean reset()
	{
		if (seed == null)
		{
			return false;
		}

		normalDistribution.reseedRandomGenerator(seed);
		return true;
	}

	@Override
	public String toString()
	{
		return "NormalDistributionInitializer{" +
				"mean=" + getMean() +
				", standard deviation=" + getStandardDeviation() +
				", seed=" + seed +
				'}';
	}
}
