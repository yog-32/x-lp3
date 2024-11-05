// Java program to solve fractional Knapsack Problem

import java.util.Arrays;
import java.util.Comparator;

// Greedy approach
public class fknapsack {
	
	// Function to get maximum value
	private static double getMaxValue(ItemValue[] arr, int capacity) {
		// Sorting items by profit/weight ratio;
		Arrays.sort(arr, new Comparator<ItemValue>() {
			@Override
			public int compare(ItemValue item1, ItemValue item2) {
				double cpr1 = (double)item1.profit / item1.weight;
				double cpr2 = (double)item2.profit / item2.weight;

				if (cpr1 < cpr2)
					return 1;
				else
					return -1;
			}
		});

		double totalValue = 0d;

		for (ItemValue i : arr) {
			int curWt = i.weight;
			int curVal = i.profit;

			if (capacity - curWt >= 0) {
				// This weight can be picked whole
				capacity -= curWt;
				totalValue += curVal;
			} else {
				// Item can't be picked whole
				double fraction = (double)capacity / curWt;
				totalValue += curVal * fraction;
				break;
			}
		}

		return totalValue;
	}

	// Item value class
	static class ItemValue {
		int profit, weight;

		// Item value function
		public ItemValue(int val, int wt) {
			this.weight = wt;
			this.profit = val;
		}
	}

	// Driver code
	public static void main(String[] args) {
		ItemValue[] arr = { new ItemValue(60, 10), new ItemValue(100, 20), new ItemValue(120, 30) };
		int capacity = 50;

		double maxValue = getMaxValue(arr, capacity);

		// Function call
		System.out.println(maxValue);
	}
}
