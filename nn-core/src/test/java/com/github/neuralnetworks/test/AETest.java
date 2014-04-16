package com.github.neuralnetworks.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.architecture.types.NNFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;

public class AETest {

    /**
     * Autoencoder backpropagation
     */
    @Test
    public void testAEBackpropagation() {
	// sequential execution for debugging
	Environment.getInstance().setExecutionMode(EXECUTION_MODE.CPU);

	// autoencoder with 6 input/output and 2 hidden units
	Autoencoder ae = NNFactory.autoencoderSigmoid(6, 2, true, true);

	// We'll use a simple dataset of symptoms of a flu illness. There are 6
	// input features and the first three are symptoms of the illness - for
	// example 1 0 0 0 0 0 means that a patient has high temperature, 0 1
	// 0 0 0 0 - coughing, 1 1 0 0 0 0 - coughing and high temperature
	// and so on. The second three features are "counter" symptomps - when a
	// patient has one of those it is less likely that he's sick. For
	// example 0 0 0 1 0 0 means that he has a flu vaccine. It's possible
	// to have combinations between both - for exmample 0 1 0 1 0 0 means
	// that the patient is vaccinated, but he's also coughing. We will
	// consider a patient to be sick when he has at least two of the first
	// three and healthy if he has two of the second three
	TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, null);
	TrainingInputProvider testInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 }, { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, new float[][] { { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } });
	MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

	// backpropagation for autoencoders
	BackPropagationAutoencoder t = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f, 0.01f)), 0.1f, 0.5f, 0f, 0f, 0f, 1, 1, 100);

	// log data
	t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

	// early stopping
	//t.addEventListener(new EarlyStoppingListener(t.getTrainingInputProvider(), 1000, 0.1f));

	// training
	t.train();

	// the output layer is removed, thus making the hidden layer the new output
	ae.removeLayer(ae.getOutputLayer());

	// testing
	t.test();

	assertEquals(0, t.getOutputError().getTotalNetworkError(), 0);
    }
}
