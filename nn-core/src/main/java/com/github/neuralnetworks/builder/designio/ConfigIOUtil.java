package com.github.neuralnetworks.builder.designio;

import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.protobuf.nn.NNProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.TrainerProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.mapping.NNMapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.mapping.TrainerMapper;

/**
 * @author tmey
 */
public class ConfigIOUtil {

	public static NeuralNetworkBuilder mapProtoBufTo(NNProtoBufWrapper.NetConfiguration netParameter,
			TrainerProtoBufWrapper.TrainParameter solverParameter) {
		NeuralNetworkBuilder neuralNetworkBuilder = new NeuralNetworkBuilder();

		TrainerMapper.mapProtoBufTrainConfigToBuilder(neuralNetworkBuilder, solverParameter);
		NNMapper.mapProtoBufNetConfigToBuilder(neuralNetworkBuilder, netParameter);

		return neuralNetworkBuilder;
	}

}
