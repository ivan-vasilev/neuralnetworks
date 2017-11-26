package com.github.neuralnetworks.util;

import java.io.Serializable;
import java.util.Arrays;

import org.apache.commons.lang.builder.HashCodeBuilder;

public class CustomArray implements Serializable
{
	private static final long serialVersionUID = 1L;

	private int[] array;

	public CustomArray(int[] array)
	{
		super();
		this.array = array;
	}

	public int[] getArray()
	{
		return array;
	}

	public void setArray(int[] array)
	{
		this.array = array;
	}

	@Override
	public boolean equals(Object obj)
	{
		if (!(obj instanceof CustomArray))
			return false;
		if (obj == this)
			return true;

		CustomArray a = (CustomArray) obj;

		return Arrays.equals(a.array, this.array);
	}

	@Override
	public int hashCode()
	{
		HashCodeBuilder hcb = new HashCodeBuilder();
		for (int i : array)
		{
			hcb.append(i);
		}

		return hcb.toHashCode();
	}
}
