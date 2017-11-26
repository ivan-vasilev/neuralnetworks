package com.github.neuralnetworks.analyzing.network.comparison;

/**
 * @author tmey
 */
public class DifferentNetworksException extends Exception
{
	private static final long serialVersionUID = 1L;

	private ComparisonResult comparisonResult = null;

	public DifferentNetworksException(String message)
	{
		super(message);
	}

	public DifferentNetworksException(String message, Throwable cause)
	{
		super(message, cause);
	}

	public DifferentNetworksException(String message, ComparisonResult comparisonResult)
	{
		super(message);
		this.comparisonResult = comparisonResult;
	}

	public DifferentNetworksException(String message, Throwable cause, ComparisonResult comparisonResult)
	{
		super(message, cause);
		this.comparisonResult = comparisonResult;
	}

	public ComparisonResult getComparisonResult()
	{
		return comparisonResult;
	}
}
