package com.github.neuralnetworks.builder.layer.structure;

import com.github.neuralnetworks.builder.activation.ActivationType;
import com.github.neuralnetworks.builder.activation.TransferFunctionType;

/**
 * @author tmey
 */
public interface MainFunctionsChangeable
{
	LayerBuilder setTransferFunctionType(TransferFunctionType transferFunctionType);

	LayerBuilder setActivationType(ActivationType activationType);
}
