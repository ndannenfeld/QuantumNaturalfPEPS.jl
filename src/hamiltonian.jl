function getIsing_H(Lx::Int64,Ly::Int64,J::Float64)
    H = ITensor()
    Sz = [1 0; 0 -1]
        
    i_H_u1 = Index(2,"ind_H_u1")
    i_H_u2 = Index(2,"ind_H_u2")
    i_H_d1 = Index(2,"ind_H_d1")
    i_H_d2 = Index(2,"ind_H_d2")

    H = ITensor(J*Sz, ind_H_u1, ind_H_d1)*ITensor(Sz, ind_H_u2, ind_H_d2)
end