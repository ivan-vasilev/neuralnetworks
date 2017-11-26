package com.github.neuralnetworks.training;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import com.github.neuralnetworks.events.TrainingEvent;
import com.github.neuralnetworks.events.TrainingEventListener;
import com.github.neuralnetworks.util.Constants;

/**
 * Provider for hyperparameters
 */
public class Hyperparameters implements Serializable, TrainingEventListener
{
	private static final long serialVersionUID = 1L;

	private Map<String, Object> defaultValues;
	private Map<Object, Map<String, Object>> objectValues;

	public Hyperparameters()
	{
		super();
		defaultValues = new HashMap<>();
		objectValues = new HashMap<>();

		setDefaultLearningRate(0);
		setDefaultMomentum(0f);
		setDefaultL1WeightDecay(0f);
		setDefaultL2WeightDecay(0f);
		setDefaultDropoutRate(0f);
	}

	@SuppressWarnings("unchecked")
	public <T> T get(Object key, String parameter)
	{
		T result = null;
		if (key != null && objectValues.containsKey(key))
		{
			result = (T) objectValues.get(key).get(parameter);
		}

		if (result == null)
		{
			result = getDefault(parameter);
		}

		return result;
	}

	@SuppressWarnings("unchecked")
	public <T> T getDefault(String parameter)
	{
		return (T) defaultValues.get(parameter);
	}

	public <T> void set(Object key, String parameter, T value)
	{
		Map<String, Object> objectValues = this.objectValues.get(key);
		if (value != null)
		{
			if (objectValues == null)
			{
				this.objectValues.put(key, objectValues = new HashMap<>());
			}
			objectValues.put(parameter, value);
		} else if (objectValues != null)
		{
			objectValues.remove(parameter);
		}
	}

	public <T> void setDefault(String parameter, T value)
	{
		if (value != null)
		{
			defaultValues.put(parameter, value);
		} else
		{
			defaultValues.remove(parameter);
		}
	}

	public float getLearningRate(Object key)
	{
		return get(key, Constants.LEARNING_RATE);
	}

	public void setLearningRate(Object key, float learningRate)
	{
		set(key, Constants.LEARNING_RATE, learningRate);
	}

	public float getDefaultLearningRate()
	{
		return getDefault(Constants.LEARNING_RATE);
	}

	public void setDefaultLearningRate(float learningRate)
	{
		setDefault(Constants.LEARNING_RATE, learningRate);
	}

	public float getMomentum(Object key)
	{
		return get(key, Constants.MOMENTUM);
	}

	public void setMomentum(Object key, float momentum)
	{
		set(key, Constants.MOMENTUM, momentum);
	}

	public float getDefaultMomentum()
	{
		return getDefault(Constants.MOMENTUM);
	}

	public void setDefaultMomentum(float momentum)
	{
		setDefault(Constants.MOMENTUM, momentum);
	}

	public float getL1WeightDecay(Object key)
	{
		return get(key, Constants.L1_WEIGHT_DECAY);
	}

	public void setL1WeightDecay(Object key, float l1WeightDecay)
	{
		set(key, Constants.L1_WEIGHT_DECAY, l1WeightDecay);
	}

	public float getDefaultL1WeightDecay()
	{
		return getDefault(Constants.L1_WEIGHT_DECAY);
	}

	public void setDefaultL1WeightDecay(float l1WeightDecay)
	{
		setDefault(Constants.L1_WEIGHT_DECAY, l1WeightDecay);
	}

	public float getL2WeightDecay(Object key)
	{
		return get(key, Constants.L2_WEIGHT_DECAY);
	}

	public void setL2WeightDecay(Object key, float l2WeightDecay)
	{
		set(key, Constants.L2_WEIGHT_DECAY, l2WeightDecay);
	}

	public float getDefaultL2WeightDecay()
	{
		return getDefault(Constants.L2_WEIGHT_DECAY);
	}

	public void setDefaultL2WeightDecay(float l2WeightDecay)
	{
		setDefault(Constants.L2_WEIGHT_DECAY, l2WeightDecay);
	}

	public float getDropoutRate(Object key)
	{
		return get(key, Constants.DROPOUT_RATE);
	}

	public void setDropoutRate(Object key, float dropoutRate)
	{
		set(key, Constants.DROPOUT_RATE, dropoutRate);
	}

	public float getDefaultDropoutRate()
	{
		return getDefault(Constants.DROPOUT_RATE);
	}

	public void setDefaultDropoutRate(float dropoutRate)
	{
		setDefault(Constants.DROPOUT_RATE, dropoutRate);
	}

	@Override
	public void handleEvent(TrainingEvent event)
	{
	}
}
