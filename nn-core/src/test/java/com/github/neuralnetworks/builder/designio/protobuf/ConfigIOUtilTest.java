package com.github.neuralnetworks.builder.designio.protobuf;

import com.github.neuralnetworks.builder.NeuralNetworkBuilder;
import com.github.neuralnetworks.builder.designio.ConfigIOUtil;
import com.github.neuralnetworks.builder.designio.protobuf.nn.InputProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.NNProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.TrainerProtoBufWrapper;
import com.github.neuralnetworks.builder.designio.protobuf.nn.mapping.NNMapper;
import com.google.protobuf.TextFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.file.Files;
import junit.framework.TestCase;
import org.apache.commons.io.IOUtils;
import org.junit.Test;

public class ConfigIOUtilTest extends TestCase {
	private File testDirectory;
	private String resource = "image1.png";

	public void createTmpInputDirectory() throws Exception {
		testDirectory = Files.createTempDirectory("test").toFile();
		testDirectory.deleteOnExit();
		File category1 = new File(testDirectory, "category1");
		category1.mkdir();
		category1.deleteOnExit();

		File targetFile = new File(category1, resource);
		targetFile.deleteOnExit();

		InputStream is = null;
		OutputStream os = null;
		try {
			is = Thread.currentThread().getContextClassLoader().getResourceAsStream("images/" + resource);
			if (is == null) {
				throw new IllegalArgumentException("Could not found resource " + resource + " inresource path");
			}
			os = new FileOutputStream(targetFile);

			IOUtils.copy(is, os);
		} finally {
			IOUtils.closeQuietly(is);
			IOUtils.closeQuietly(os);
		}
	}

	@Test
	public void testMapProtoBufNetConfigToBuilder() throws Exception {

		InputStreamReader reader = new InputStreamReader(
				ClassLoader.getSystemResourceAsStream("nn/nnconfiguration.prototxt"));

		NNProtoBufWrapper.NetConfiguration.Builder builder = NNProtoBufWrapper.NetConfiguration.newBuilder();
		TextFormat.merge(reader, builder);
		NNProtoBufWrapper.NetConfiguration netConfiguration = builder.build();

		NeuralNetworkBuilder neuralNetworkBuilder = NNMapper.mapProtoBufNetConfigToBuilder(netConfiguration);

		System.out.println(neuralNetworkBuilder.toString());

	}

	@Test
	public void testReadProtoBufTrainer() throws Exception {

		InputStreamReader reader = new InputStreamReader(ClassLoader.getSystemResourceAsStream("nn/training.prototxt"));

		TrainerProtoBufWrapper.TrainParameter.Builder builder = TrainerProtoBufWrapper.TrainParameter.newBuilder();
		TextFormat.merge(reader, builder);
	}

	@Test
	public void test() throws Exception {
		createTmpInputDirectory();

		TrainerProtoBufWrapper.TrainParameter trainer;
		NNProtoBufWrapper.NetConfiguration netConfiguration;

		{
			InputStreamReader reader = new InputStreamReader(
					ClassLoader.getSystemResourceAsStream("nn/nnconfiguration.prototxt"));

			NNProtoBufWrapper.NetConfiguration.Builder builder = NNProtoBufWrapper.NetConfiguration.newBuilder();
			TextFormat.merge(reader, builder);
			netConfiguration = builder.build();
		}

		{
			InputStreamReader reader = new InputStreamReader(
					ClassLoader.getSystemResourceAsStream("nn/training.prototxt"));

			TrainerProtoBufWrapper.TrainParameter.Builder builder = TrainerProtoBufWrapper.TrainParameter.newBuilder();
			TextFormat.merge(reader, builder);

			// replace path
			{
				TrainerProtoBufWrapper.TrainParameter.Builder replaceBuilder = TrainerProtoBufWrapper.TrainParameter
						.newBuilder()
						.setTrainInput(
								InputProtoBufWrapper.InputData.newBuilder().setPath(testDirectory.getAbsolutePath()))
						.setTestParam(TrainerProtoBufWrapper.TestParameter.newBuilder().setTestInput(
								InputProtoBufWrapper.InputData.newBuilder().setPath(testDirectory.getAbsolutePath())))
						.setValidationParam(TrainerProtoBufWrapper.ValidationParameter.newBuilder().setValidationInput(
								InputProtoBufWrapper.InputData.newBuilder().setPath(testDirectory.getAbsolutePath())));

				builder.mergeFrom(replaceBuilder.build());
			}

			trainer = builder.build();
		}

		NeuralNetworkBuilder neuralNetworkBuilder = ConfigIOUtil.mapProtoBufTo(netConfiguration, trainer);

		System.out.println(neuralNetworkBuilder.toString());
	}
}