function hamiltonain_J1J2(J1, J2, Lx, Ly; operators=["X", "Y", "Z"])
    ham_J1J2 = OpSum()
    for i in 1:Lx, j in 1:Ly, t in operators
        # J1
        if j != Ly
            ham_J1J2 += (J1, t, (i,j), t, (i,j+1))
        end
        if i != Lx
            ham_J1J2 += (J1, t, (i,j), t, (i+1,j))
        end

        # J2
        if i != Lx && j != Ly
            ham_J1J2 += (J2, t, (i,j), t, (i+1,j+1))
            ham_J1J2 += (J2, t, (i+1,j), t, (i,j+1))
        end
    end
    return ham_J1J2
end