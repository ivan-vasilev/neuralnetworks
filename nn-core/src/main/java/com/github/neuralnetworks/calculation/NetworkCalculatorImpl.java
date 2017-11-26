package com.github.neuralnetworks.calculation;

import com.github.neuralnetworks.architecture.NeuralNetwork;
import com.github.neuralnetworks.util.Properties;

/**
 * 
 * Default implementation of NetworkCalculator
 *
 * @param <N>
 */
public class NetworkCalculatorImpl<N extends NeuralNetwork> implements NetworkCalculator<N>
{
	private static final long serialVersionUID = 1L;

	protected Properties properties;

	protected transient boolean skipCurrentMiniBatch;

	public NetworkCalculatorImpl(Properties properties)
	{
		super();
		this.properties = properties;
	}

	@Override
	public Properties getProperties()
	{
		return properties;
	}

	@Override
	public void setSkipCurrentMiniBatch(boolean skipCurrentMiniBatch)
	{
		this.skipCurrentMiniBatch = skipCurrentMiniBatch;
	}

	@Override
	public boolean getSkipCurrentMiniBatch()
	{
		return skipCurrentMiniBatch;
	}
}
