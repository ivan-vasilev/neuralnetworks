package com.github.neuralnetworks.samples.test;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public class Test {
    public static void main(String[] args) {

	final float[] a = new float[5];

	Kernel k = new Kernel() {

	    @Override
	    public void run() {
		int id = getGlobalId();
		a[id] = id;
	    }
	};

	k.setExecutionMode(EXECUTION_MODE.GPU);
	k.setExplicit(true);

	k.execute(a.length);

	System.out.println();
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

	private float[] arr2;

	public TestKernel() {
	    super();
	    this.arr = new float[5];
	    this.arr2 = new float[5];
	    for (int i = 0; i < arr.length; i++) {
		arr[i] = i + 1;
	    }
	}

	public void calc() {
	    setExecutionMode(EXECUTION_MODE.GPU);
	    setExplicit(true);
	    put(arr);
	    put(arr2);
	    execute(5);
	    get(arr);
	    get(arr2);
	}

	@Override
	public void run() {
	    int i = getGlobalId();
	    arr2[i] = arr[i];
	}
    }

    public static class TestKernel2 extends Kernel {

	private float[] arr = new float[5];

	@Local
	private final float[] x = new float[] { 5 };

	@Override
	public void run() {
	    int i = getGlobalId();
	    arr[i] = i + x[0];
	}
    }
}
