package com.github.neuralnetworks.samples.test;

import com.amd.aparapi.Kernel;


public class Test {
    public static void main(String[] args) {
	int x = fact(5);
	System.out.println(x);

    }

    public static int fact(int n) {
	if (n == 0) {
	    return 1;
	}

	int result = n * fact(n - 1);
	return result;
    }

    public static class TestKernel extends Kernel {

	private float[] arr;
	private final float x = 5;

	@Override
	public void run() {
	    int i = getGlobalId();
	    arr[i] = i + x;
	}
    }
    
    public static class TestKernel2 extends Kernel {
	
	private float[] arr;

	@Local
	private final float[] x = new float[] {5};
	
	@Override
	public void run() {
	    int i = getGlobalId();
	    arr[i] = i + x[0];
	}
    }
}
