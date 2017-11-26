package com.github.neuralnetworks.training.random;

import com.github.neuralnetworks.tensor.Tensor;

/**
 * @author Urko
 */
public class DummyFixedInitializer implements RandomInitializer
{

	private static final long serialVersionUID = 1L;

	private final float value;

	public DummyFixedInitializer()
	{
		value = 0.01f;
	}

	public DummyFixedInitializer(float value)
	{
		this.value = value;
	}

	@Override
    public void initialize(Tensor t) {
        float[] elements = t.getElements();
        t.forEach(i -> elements[i] = value);
    }

	public boolean reset()
	{
		return true;
	}

	@Override
	public String toString()
	{
		return "DummyFixedInitializer{" +
				"value=" + value +
				'}';
	}
}
