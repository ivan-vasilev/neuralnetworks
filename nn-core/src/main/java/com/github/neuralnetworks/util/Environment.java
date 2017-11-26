package com.github.neuralnetworks.util;

/**
 * Singleton for environment variables (can be used for debugging)
 */
public class Environment
{

	private static Environment singleton = new Environment();

	private RuntimeConfiguration runtimeConfiguration;

	private Environment()
	{
		runtimeConfiguration = new RuntimeConfiguration();
	}

	public static Environment getInstance()
	{
		return singleton;
	}

	public RuntimeConfiguration getRuntimeConfiguration()
	{
		return runtimeConfiguration;
	}

	public void setRuntimeConfiguration(RuntimeConfiguration runtimeConfiguration)
	{
		this.runtimeConfiguration = runtimeConfiguration;
	}
}
