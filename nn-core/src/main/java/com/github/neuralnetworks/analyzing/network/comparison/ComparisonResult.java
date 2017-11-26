package com.github.neuralnetworks.analyzing.network.comparison;

import java.util.Map;

import com.github.neuralnetworks.architecture.Connections;

/**
 * @author tmey
 */
public class ComparisonResult
{
	private Map<Object, Float> mapActivationDistance;
	private Map<Object, Float> mapBackPropagationActivationDistance;
	private Map<Connections, Float> mapWeightDistance;

	public Map<Object, Float> getMapActivationDistance()
	{
		return mapActivationDistance;
	}

	public void setMapActivationDistance(Map<Object, Float> mapActivationDistance)
	{
		this.mapActivationDistance = mapActivationDistance;
	}

	public Map<Object, Float> getMapBackPropagationActivationDistance()
	{
		return mapBackPropagationActivationDistance;
	}

	public void setMapBackPropagationActivationDistance(Map<Object, Float> mapBackPropagationActivationDistance)
	{
		this.mapBackPropagationActivationDistance = mapBackPropagationActivationDistance;
	}

	public Map<Connections, Float> getMapWeightDistance()
	{
		return mapWeightDistance;
	}

	public void setMapWeightDistance(Map<Connections, Float> mapWeightDistance)
	{
		this.mapWeightDistance = mapWeightDistance;
	}
}
