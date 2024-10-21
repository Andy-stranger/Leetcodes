namespace LeetCodes.Sorting
{
    public static class Sort
    {
        public static void SelectionSort(this int[] array)
        {
            try
            {
                int n = array.Length;
                for (int i = 0; i < n; i++)
                {
                    int minInd = i;
                    for (int j = i; j < n; j++)
                    {
                        if (array[j] < array[minInd]) minInd = j;
                    }
                    (array[i], array[minInd]) = (array[minInd], array[i]);
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
            
        }

        public static void BubbleSort(this int[] array)
        {
            try
            {
                int n = array.Length;
                for (int i = n-1; i >= 0; i--)
                {
                    bool isSorted = true;
                    for (int j = 0; j <= i-1; j++)
                    {
                        if (array[j] > array[j + 1])
                        {
                            (array[j], array[j + 1]) = (array[j + 1], array[j]);
                            isSorted = false;
                        }
                    }
                    if (isSorted) break;
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message); 
                throw;
            }
            
        }

        public static void InsertionSort(this int[] array)
        {
            try
            {
                int n = array.Length;
                for (int i = 0; i < n; i++)
                {
                    int j = i;
                    while (j > 0 && array[j - 1] > array[j])
                    {
                        (array[j], array[j - 1]) = (array[j - 1], array[j]);
                        j--;
                    }
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
            
        }

        public static void MergeSort(this int[] array, int l, int r)
        {
            try
            {
                if (l < r)
                {
                    int mid = (l + r) / 2;
                    array.MergeSort(l, mid);
                    array.MergeSort(mid + 1, r);
                    Merge(array, l, mid, r);
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message); 
                throw;
            }
            
        }

        private static void Merge(int[] array, int l, int m, int r)
        {
            try
            {
                int i = 0;
                int[] temp = new int[array.Length];
                int left = l;
                int right = m + 1;
                while (left <= m && right <= r)
                {
                    if (array[left] <= array[right])
                    {
                        temp[i] = array[left];
                        left++;
                    }
                    else
                    {
                        temp[i] = array[right];
                        right++;
                    }
                    i++;
                }
                while (left <= m)
                {
                    temp[i] = array[left];
                    left++;
                    i++;
                }
                while (right <= r)
                {
                    temp[i] = array[right];
                    right++;
                    i++;
                }
                for (int k = l; k <= r; k++)
                {
                    array[k] = temp[k - l];
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
            
        }

        public static void QuickSort(this int[] array, int l, int r)
        {
            try
            {
                if (l < r)
                {
                    int pivotIndex = Partition(array, l, r);
                    array.QuickSort(l, pivotIndex - 1);
                    array.QuickSort(pivotIndex + 1, r);
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
            
        }

        private static int Partition(int[] array, int l, int r)
        {
            try
            {
                int pivot = array[l];
                int left = l;
                int right = r;
                while (left < right)
                {
                    while (array[left] <= pivot && left <= r - 1)
                    {
                        left++;
                    }
                    while (array[right] > pivot && right >= l + 1)
                    {
                        right--;
                    }
                    if(left < right)
                    {
                        (array[left], array[right]) = (array[right], array[left]);
                    }
                }
                (array[l], array[right]) = (array[right], array[l]);
                return right;

            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.Message);
                throw;
            }
        }
    }
}
