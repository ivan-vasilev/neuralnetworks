package com.github.neuralnetworks.builder.designio;

import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.protobuf.ProtoBufNNConfigMapper;
import com.github.neuralnetworks.builder.designio.protobuf.ProtoBufWrapper;
import com.google.protobuf.TextFormat;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * @author tmey
 */
public class CaffeConfigIOUtil
{

	public static NeuralNetworkBuilder readBuilderFromClasspath(String networkConfigFileName, String trainingConfigFileName) throws IOException
	{
		return readBuilderFromStream(ClassLoader.getSystemResourceAsStream(networkConfigFileName), ClassLoader.getSystemResourceAsStream(trainingConfigFileName));
	}

	public static NeuralNetworkBuilder readBuilderFromFile(File networkConfigFile, File trainingConfigFile) throws IOException
	{
		return readBuilderFromStream(new FileInputStream(networkConfigFile), new FileInputStream(trainingConfigFile));
	}

	public static NeuralNetworkBuilder readBuilderFromStream(InputStream networkConfigStream, InputStream trainingConfigStream) throws IOException
	{

		// parse net configuration
		InputStreamReader reader = new InputStreamReader(networkConfigStream);

		ProtoBufWrapper.NetParameter.Builder builder = ProtoBufWrapper.NetParameter.newBuilder();
		TextFormat.merge(reader, builder);
		ProtoBufWrapper.NetParameter netParameter = builder.build();

		// parse solver configuration
		reader = new InputStreamReader(trainingConfigStream);

		ProtoBufWrapper.SolverParameter.Builder solverBuilder = ProtoBufWrapper.SolverParameter.newBuilder();
		TextFormat.merge(reader, solverBuilder);
		ProtoBufWrapper.SolverParameter solverParameter = solverBuilder.build();

		return ProtoBufNNConfigMapper.mapProtoBufTo(netParameter, solverParameter);
	}

}
