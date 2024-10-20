using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    internal class MergeSort
    {
        private void Merge(int[] array, int left, int mid, int right)
        {
            int[] temp = new int[right - left + 1];
            int l = left;
            int r = mid + 1;
            int i = 0;
            while (l <= mid && r <= right)
            {
                if (array[l] < array[r])
                {
                    temp[i] = array[l];
                    l++;
                }
                else
                {
                    temp[i] = array[r];
                    r++;
                }
                i++;
            }

            while (l <= mid)
            {
                temp[i] = array[l];
                l++;
                i++;
            }

            while (r <= right)
            {
                temp[i] = array[r];
                r++;
                i++;
            }

            for (i = 0; i < temp.Length; i++)
            {
                array[left + i] = temp[i];
            }
        }
        public void Sort(int[] array, int left, int right)
        {
            if (left < right)
            {
                int mid = (right + left) / 2;
                Sort(array, left, mid);
                Sort(array, mid + 1, right);
                Merge(array, left, mid, right);
            }
        }
    }
}
