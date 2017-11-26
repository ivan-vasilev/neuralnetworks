package com.github.neuralnetworks.architecture.types;

import com.amd.aparapi.Kernel;
import com.github.neuralnetworks.architecture.types.Autoencoder;
import com.github.neuralnetworks.calculation.CalculationFactory;
import com.github.neuralnetworks.input.MultipleNeuronsOutputError;
import com.github.neuralnetworks.input.SimpleInputProvider;
import com.github.neuralnetworks.test.AbstractTest;
import com.github.neuralnetworks.training.TrainerFactory;
import com.github.neuralnetworks.training.TrainingInputProvider;
import com.github.neuralnetworks.training.backpropagation.BackPropagationAutoencoder;
import com.github.neuralnetworks.training.events.LogTrainingListener;
import com.github.neuralnetworks.training.random.MersenneTwisterRandomInitializer;
import com.github.neuralnetworks.training.random.NNRandomInitializer;
import com.github.neuralnetworks.util.Environment;
import com.github.neuralnetworks.util.RuntimeConfiguration;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by chass on 02.12.14.
 */
@RunWith(Parameterized.class)
public class AutoencoderTests extends AbstractTest {


    public AutoencoderTests(RuntimeConfiguration conf)
    {
        Environment.getInstance().setRuntimeConfiguration(conf);
    }

    @Parameterized.Parameters
    public static Collection<RuntimeConfiguration[]> runtimeConfigurations()
    {
        List<RuntimeConfiguration[]> configurations = new ArrayList<>();

        RuntimeConfiguration conf1 = new RuntimeConfiguration();
        conf1.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
        conf1.setUseDataSharedMemory(false);
        conf1.setUseWeightsSharedMemory(false);
        configurations.add(new RuntimeConfiguration[] { conf1 });

        RuntimeConfiguration conf2 = new RuntimeConfiguration();
        conf2.getAparapiConfiguration().setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
        conf2.setUseDataSharedMemory(true);
        conf2.setUseWeightsSharedMemory(true);
        configurations.add(new RuntimeConfiguration[] { conf2 });

//		RuntimeConfiguration conf3 = new RuntimeConfiguration();
//		conf3.setCalculationProvider(CalculationProvider.OPENCL);
//		conf3.setUseDataSharedMemory(false);
//		conf3.setUseWeightsSharedMemory(false);
//		conf3.getOpenCLConfiguration().setAggregateOperations(false);
//		conf3.getOpenCLConfiguration().setSynchronizeAfterOpertation(true);
//		conf3.getAparapiConfiguration().setExecutionMode(EXECUTION_MODE.SEQ);
//		configurations.add(new RuntimeConfiguration[] { conf3 });

        return configurations;
    }

    /**
     * Autoencoder backpropagation
     */
    @Test
    @Ignore // issue with LagTrainingListener
    public void testAEBackpropagation()
    {
        // autoencoder with 6 input/output and 2 hidden units
        Autoencoder ae = CalculationFactory.autoencoderSigmoid(6, 2, true);

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
        TrainingInputProvider trainInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 },
                { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } });
        TrainingInputProvider testInputProvider = new SimpleInputProvider(new float[][] { { 1, 1, 1, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 }, { 1, 1, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0, 0 }, { 0, 1, 1, 1, 0, 0 },
                { 0, 0, 0, 1, 1, 1 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 }, { 0, 0, 0, 0, 1, 1 }, { 0, 0, 0, 1, 1, 0 } }, new float[][] { { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 1, 0 }, { 0, 1 },
                { 0, 1 }, { 0, 1 }, { 0, 1 }, { 0, 1 } });
        MultipleNeuronsOutputError error = new MultipleNeuronsOutputError();

        // backpropagation for autoencoders
        BackPropagationAutoencoder t = TrainerFactory.backPropagationAutoencoder(ae, trainInputProvider, testInputProvider, error, new NNRandomInitializer(new MersenneTwisterRandomInitializer(-0.01f,
                0.01f)), 0.02f, 0.7f, 0f, 0f, 0.00001f, 1, 1, 200);

        // log data
        t.addEventListener(new LogTrainingListener(Thread.currentThread().getStackTrace()[1].getMethodName(), true, false));

        // early stopping
        // t.addEventListener(new EarlyStoppingListener(t.getTrainingInputProvider(), 1000, 0.1f));

        // training
        t.train();

        // the output layer is removed, thus making the hidden layer the new output
        ae.removeLayer(ae.getOutputLayer());

        // testing
        t.test();

        assertEquals(0, t.getOutputError().getTotalNetworkError(), 0.1);
    }
}
