#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        auto shapeA = inputs[0]->getDims();
        auto shapeB = inputs[1]->getDims();
        
        int rankA = shapeA.size();
        int rankB = shapeB.size();
        
        IT_ASSERT(rankA >= 2 && rankB >= 2);
        
        int m = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];
        int kA = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
        int kB = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        int n = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];
        
        IT_ASSERT(kA == kB);
        
        Shape batchDimsA, batchDimsB;
        for (int i = 0; i < rankA - 2; ++i) {
            batchDimsA.push_back(shapeA[i]);
        }
        for (int i = 0; i < rankB - 2; ++i) {
            batchDimsB.push_back(shapeB[i]);
        }
        
        Shape batchDims = infer_broadcast(batchDimsA, batchDimsB);
        
        Shape result = batchDims;
        result.push_back(m);
        result.push_back(n);
        
        this->m = m;
        this->n = n;
        this->k = kA;
        
        return {{result}};
    }

} // namespace infini