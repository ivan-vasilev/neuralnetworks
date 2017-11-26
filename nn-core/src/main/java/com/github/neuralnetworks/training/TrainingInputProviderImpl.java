package com.github.neuralnetworks.training;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.github.neuralnetworks.calculation.operations.TensorFunction;
import com.github.neuralnetworks.input.InputConverter;

public abstract class TrainingInputProviderImpl implements TrainingInputProvider
{
	private static final long serialVersionUID = 1L;

	private static final Logger logger = LoggerFactory.getLogger(TrainingInputProviderImpl.class);

	/**
	 * List of modifiers to apply on the input data after the conversion
	 */
	private List<TensorFunction> inputModifiers;

	/**
	 * Converter for the target
	 */
	private InputConverter targetConverter;

	/**
	 * Counter
	 */
	protected int currentInput;

	/**
	 * compares the hashes of sequential inputs by checking for irregular values
	 */
	private boolean compareInputs;

	private Set<String> hashes;

	public TrainingInputProviderImpl()
	{
		super();
	}

	public TrainingInputProviderImpl(InputConverter targetConverter)
	{
		super();
		this.targetConverter = targetConverter;
	}

	@Override
	public void populateNext(TrainingInputData ti)
	{
		TrainingInputProvider.super.populateNext(ti);

		// a simple check that tries to verify if the input data is correct - it compares if there are equivalent inputs or if the input contains only 0
		if (compareInputs)
		{
			if (hashes == null)
			{
				hashes = new HashSet<>();
			}

			boolean containsNonZero = false;
			int mb = ti.getInput().getDimensions()[0];
			for (int i = 0; i < mb; i++)
			{
				byte[] hashArr = new byte[4 * (ti.getInput().getSize() / mb)];
				for (int j = 0; j < ti.getInput().getSize() / mb; j++)
				{
					float v = ti.getInput().getElements()[ti.getInput().getStartIndex() + i * (ti.getInput().getSize() / mb) + j];
					if (v != 0)
					{
						containsNonZero = true;
					}
					byte[] b = ByteBuffer.allocate(4).putFloat(v).array();
					for (int k = 0; k < b.length; k++)
					{
						hashArr[4 * j + k] = b[k];
					}
				}

				String h = null;
				try
				{
					h = Arrays.toString(MessageDigest.getInstance("MD5").digest(hashArr));
				} catch (NoSuchAlgorithmException e)
				{
					e.printStackTrace();
				}

				if (!containsNonZero)
				{
					logger.warn("Input contains only 0");
				}

				if (hashes.contains(h))
				{
					logger.warn("There was already such input");
				}

				hashes.add(h);
			}
		}
	}

	@Override
	public List<TensorFunction> getInputModifiers()
	{
		return inputModifiers;
	}

	public void addInputModifier(TensorFunction modifier)
	{
		if (inputModifiers == null)
		{
			inputModifiers = new ArrayList<>();
		}

		inputModifiers.add(modifier);
	}

	public void removeModifier(TensorFunction modifier)
	{
		if (inputModifiers != null)
		{
			inputModifiers.remove(modifier);
		}
	}

	public InputConverter getTargetConverter()
	{
		return targetConverter;
	}

	public void setTargetConverter(InputConverter targetConverter)
	{
		this.targetConverter = targetConverter;
	}

	public boolean getCompareHashes()
	{
		return compareInputs;
	}

	public void setCompareHashes(boolean compareHashes)
	{
		this.compareInputs = compareHashes;
	}

	@Override
	public void beforeBatch(TrainingInputData ti)
	{
	}

	@Override
	public void afterBatch(TrainingInputData ti)
	{
		currentInput += ti.getInput().getDimensions()[0];
	}

	@Override
	public void reset()
	{
		currentInput = 0;
	}
}
