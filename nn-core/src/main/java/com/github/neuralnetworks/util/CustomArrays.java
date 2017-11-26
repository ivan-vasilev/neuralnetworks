package com.github.neuralnetworks.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class CustomArrays
{
	private static CustomArrays singleton = new CustomArrays();

	private Map<Object, Set<CustomArray>> customArrays = new HashMap<>();

	public static CustomArrays getInstance()
	{
		return singleton;
	}

	public Map<Object, Set<CustomArray>> getCustomArrays()
	{
		return customArrays;
	}
}
