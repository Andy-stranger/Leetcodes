using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetCodes
{
    public class DisjointSetBySize
    {
        public int[] parentArr;
        public int[] sizeArr;
        public int size;
        public DisjointSetBySize(int n)
        {
            this.parentArr = new int[n + 1];
            this.sizeArr = new int[n + 1];
            this.size = n;
            for (var i = 0; i <= n; i++)
            {
                this.parentArr[i] = i;
                this.sizeArr[i] = 1;
            }
        }

        public int GetNumberOfConnectedComponents(bool isZeroBased)
        {
            var start = isZeroBased ? 0 : 1;
            var end = isZeroBased ? size : size + 1;
            var count = 0;
            for (int i = start; i < end; i++)
            {
                if (this.parentArr[i] == i) count++;
            }
            return count;
        }

        public int Find(int X)
        {
            if (X == this.parentArr[X]) return X;
            this.parentArr[X] = Find(this.parentArr[X]);
            return this.parentArr[X];
        }

        public void UnionSetBySize(int X, int Z)
        {
            var upX = Find(X);
            var upZ = Find(Z);
            if (upX == upZ) return;
            if (this.sizeArr[upX] < this.sizeArr[upZ])
            {
                this.parentArr[upX] = upZ;
                this.sizeArr[upZ] += this.sizeArr[upX];
            }
            else
            {
                this.parentArr[upZ] = upX;
                this.sizeArr[upX] += this.sizeArr[upZ];
            }
        }
    }
}
