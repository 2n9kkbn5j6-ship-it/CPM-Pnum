# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:26:57 2025

@author: 33782
"""

import numpy as np 
import matplotlib.pyplot as plt
import numba
# from skimage.segmentation import find_boundaries

'''
On notera:
    type 1: cellules proliférantes
    type 2: cellules rien
    type 3: cellules nécrotiques
    
Pour l'instant, on considère que ce sont toutes le même type'
'''
#%%

ratio_air_vol_lim = 0.6
N = 100
conso_nutr = 0.02
Diff = 1.0
h = 1.0  # taille d'un pixel, pour l'équation de diffusion
S_outside = 2.0   # Nutriements en dehors de la tumeur
diff_tol = 1e-5
max_diff_iters = 5000
T_boltz = 1
J_medium_tumor = 16
lbda = 0.5
Vol_cible_init = 170
Nq = 0.5

J_11 = 20
J_10 = 1

#%%
#Je vais séparer en 3 matrices, pour mieux me repérer

'''
Type_map est la matrice avec les types de chaque cellules
num_map la matrice avec le numéro de chaque cellule
nutr_map la matrice des nutriments
'''

type_map = np.zeros((N, N), dtype=np.int32)
num_map = np.zeros((N, N), dtype=np.int32)
nutr_map = np.ones((N, N), dtype=np.float64) * S_outside
Vol_cible_list = [Vol_cible_init]  # vol cible de la première cellule


compteur_cell = 0  #Compteur cell aura len(types) contenant le nb de chaque type dedans


#%%

# @numba.njit
def init():
    cx, cy = N // 2, N // 2
    # num_cell = 1
    rr = 10
    for i in range(-rr, rr+1):
        for j in range(-rr, rr+1):
            x, y = cx+i, cy+j
            if (i*i + j*j) <= rr*rr:
                type_map[x,y] = 1
                num_map[x,y] = 1
    
    

init()

plt.imshow(type_map)
plt.show()



#%%

# @numba.njit
def centre_mass_tumeur(type_map):
    '''
    renvoie le centre de masse de la tumeur
    '''
    compteur = 0
    numerateur = np.zeros(2)
    for i in range(N):
        for j in range(N):
            if type_map[i,j] != 0:
                numerateur[0] += i
                numerateur[1] += j
                compteur += 1    
                
    return numerateur/compteur
    
# @numba.njit
def centre_mass_cell(num_map, num_cell):
    compteur = 0
    numerateur = np.zeros(2)
    for i in range(N):
        for j in range(N):
            if num_map[i,j] == num_cell:
                numerateur[0] += i
                numerateur[1] += j
                compteur += 1
    if compteur == 0:
        return numerateur
    return numerateur / compteur


#%%

# @numba.njit
def aire_cellule_i(num_map, cell_num):
    aire = 0
    for i in range(N):
        for j in range(N):
            if num_map[i,j] == cell_num:
                aire += 1
    return aire



# @numba.njit
def perimetre_cellule_i(num_map, num_cell):
    
    perimetre = 0
    
    for i in range(N):
        for j in range(N):
            if num_map[i,j] == num_cell:
                voisins = [(i+1,j), (i-1,j), (i,j+1), (i,j-1)] #A EAJOUTER : CLP !!!!!! PAS IMPORTAT POUR INSTANNT...
                for ni, nj in voisins:
                    if num_map[ni, nj] != num_cell:
                        perimetre += 1
                        break  # un voisin différent suffit pour compter le pixel comme bord
    return perimetre


#%%
# @numba.njit
def nutriment_cell_i(nutr_map, num_map, cell_num):
    '''
    renvoie la moyenne des nutriments de la cellule i
    '''
    summ, compt = 0, 0
    for i in range(N):
        for j in range(N):
            if num_map[i,j] == cell_num:
                summ += nutr_map[i,j]
                compt += 1
    if compt == 0:
        return 0.0
    
    return summ / compt


#%%

# @numba.njit
def diffusion_nutroment_solve(c, num_map, type_map):
    
    lamb = np.zeros((N, N), dtype=np.float64)   #Tableau de notre consommation locale
    
    for i in range(N):
        for j in range(N):
            if type_map[i,j] == 1:  # Cellule proliférante
                lamb[i,j] = conso_nutr
            else:
                lamb[i,j] = 0.0         #A améliorer avec des cellules quiescentes après

    # Conditions de dirichlet => S_outside partout si lambda = 0 NOOOOON
    for i in range(N):
        c[i,0] = S_outside           #CONDITIONS AU BORD à REVOIR!!!!!
        c[i,N-1] = S_outside
    for j in range(N):
        c[0,j] = S_outside
        c[N-1,j] = S_outside

    denom_const = Diff / (h**2)
    
    
    for it in range(max_iters):     #Schéma Gauss–Seidel, à revoir
        maxdiff = 0.0
        for i in range(1, N-1):
            for j in range(1, N-1):  #On évite les bords
            
                num = c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1]
                denom = 4.0 + lamb[i,j]/denom_const
                c_new = num / denom  
                
                diff = abs(c_new - c[i,j])
                if diff > maxdiff:
                    maxdiff = diff
                c[i,j] = c_new
                
        if maxdiff < tol:
            break
        
    return c

max_iters = 100
tol = 2
# type_map = np.zeros((N, N), dtype=np.int32)
# num_map = np.zeros((N, N), dtype=np.int32)
# nutr_map = np.ones((N, N), dtype=np.float64) * S_outside
# init()
# for i in range(10000):
#     nutr_map = diffusion_nutroment_solve(nutr_map, num_map, type_map)
#     if i%100 == 0:
#         plt.imshow(nutr_map)
#         plt.show()

#%%

# @numba.njit
def Growth_rate(Nutriment, Nq):
    if Nutriment < Nq:
        return 0
    elif Nutriment < 3 * Nq:
        return 1/2 * (1 - Nutriment/Nq)**2
    return 2    


# @numba.njit  
def increase_vol_cible(c, num_map, type_map, num_cell, Vol_cible_list):
    
    Nutr = nutriment_cell_i(nutr_map, num_map, num_cell)
    Vol_cible_list[num_cell - 1] += Growth_rate(Nutr, Nq) * Vol_cible_list[num_cell - 1]
    
    return Vol_cible_list



# X = np.linspace(0,10,1000)
# G = []
# for x in X:
#     G.append(Growth_rate(x, 3))
# plt.plot(X, G)
# plt.show()

# init()
# VVVV = increase_vol_cible(nutr_map, num_map, type_map, 1, Vol_cible_list)
# print(VVVV)






#%%


aire_max = 150  #ritère à revoir...


# @numba.njit       #VERIFER OIURQUOI NULBA NE LARCHE PAS
def dois_je_me_diviser(num_map, num_cell):
    
    aire_cell = aire_cellule_i(num_map, num_cell)
    # perimetre_cell = perimetre_cellule_i(num_map, num_cell)
    
    # ratio =  perimetre_cell / aire_cell
    # print(ratio)
    # print(aire_cell)
    # print(perimetre_cell)
    
    if aire_cell >= aire_max:
        return True
    
    return False
    

print(dois_je_me_diviser(num_map, 1))

#%%



# @numba.njit
def division_cell_axis(num_map, num_cell, Vol_cible_list, compteur_cell):
    """
    Divise la cellule i en deux selon un axe qui passe par son CM (meilleure manière que j'ai trouv, à raffiner')

    """
    
    CM = centre_mass_cell(num_map, num_cell)
    next_id = np.max(num_map) + 1       #OPTIMISABLE
    
    for i in range(N):
        for j in range(N):
            if num_map[i,j] == num_cell:
                if i > CM[0]:
                    num_map[i,j] = next_id
                    
    Vol_cible_list[num_cell - 1] = aire_cellule_i(num_map, cell_num)
    Vol_cible_list[num_cell - 1] = Vol_cible_init
    Vol_cible_list.append(Vol_cible_init)
    
    compteur_cell += 1

    return num_map, Vol_cible_list, compteur_cell

# init()
# num_map, Vol_cible_list, compteur_cell = division_cell_axis(num_map, 1, Vol_cible_list, compteur_cell)
# plt.imshow(num_map)
# plt.colorbar()
# plt.show()


#%%
# Définition de la matrice des j
# for i in range(1, n + 1):
#     J_matrix[i, 0] = J_10
#     J_matrix[0, i] = J_10
#     for j in range(1, n + 1):
#         if i != j:
#             J_matrix[i, j] = J_11

# # @numba.njit
# def J(type_cell, num_cell, type_voisin, num_voisin):
#     type_cell, num_cell, type_voisin, num_voisin = int(type_cell), int(num_cell), int(type_voisin), int(num_voisin)
    
#     if num_cell != num_voisin:
#         return J_matrix[type_cell, type_voisin]
    
#     return 0



#%%

voisins_numero = np.array([(1,0),(-1,0),(0,1),(0,-1), (1,1), (-1,-1), (1,-1), (-1,1)])

# @numba.njit
def dE_surf_local(type_map, num_map,i,j, type_old, num_old, type_new, num_new):
    dE_int = 0
    for (dx, dy) in voisins_numero: 
        xi, xj = (i + dx) % N, (j + dy) % N #Les indices, avec CLP
        type_voisin, num_voisin = type_map, num_map
        E_avant = 5
        E_apres = 10
        
        dE_int += (E_apres - E_avant)
    
    return dE_int


# @numba.njit
def dE_volume_local(V_c, V_cible, type_old, num_old, type_new, num_new):
    dE = 0.0

    if num_old > 0:
        if type_old == 1:
            idx_old = num_old - 1
        else:
            idx_old = n1 + num_old - 1
        V_before = V_c[idx_old]
        dE += (V_before - 1 - V_cible[idx_old])**2 - (V_before - V_cible[idx_old])**2

    if num_new > 0:
        if type_new == 1:
            idx_new = num_new - 1
        else:
            idx_new = n1 + num_new - 1
        V_before = V_c[idx_new]
        dE += (V_before + 1 - V_cible[idx_new])**2 - (V_before - V_cible[idx_new])**2

    return lbda * dE
    




def J(type_cell, num_cell, type_voisin, num_voisin):
    type_cell, num_cell, type_voisin, num_voisin = int(type_cell), int(num_cell), int(type_voisin), int(num_voisin)
    
    if num_cell != num_voisin:
        return J_matrix[type_cell, type_voisin]
    
    return 0
    
    
    
#%%

#%% Execution
type_map = np.zeros((N, N), dtype=np.int32)
num_map = np.zeros((N, N), dtype=np.int32)
nutr_map = np.ones((N, N), dtype=np.float64) * S_outside
Vol_cible_list = [Vol_cible_init]  # vol cible de la première cellule


init()
compteur_cell = 1  #nb de cellules

#%%
lbda = 10

for aaaaaa in range(10000):

    # nutr_map = diffusion_nutroment_solve(nutr_map, num_map, type_map)
    
    for n in range(1,compteur_cell+1):
        # Vol_cible_list =  increase_vol_cible(nutr_map, num_map, type_map, n, Vol_cible_list)
        # print(Vol_cible_list)
        # print(dois_je_me_diviser(num_map, n))
        if dois_je_me_diviser(num_map, n) :
            print('DIIIIV')
            num_map, Vol_cible_list, compteur_cell = division_cell_axis(num_map, n, Vol_cible_list, compteur_cell) 
        # print(compteur_cell)
        
    voisins_numero = [(1,0),(-1,0),(0,1),(0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    
    J_matrix = np.zeros((compteur_cell + 1, compteur_cell + 1))

    # Définition de la matrice des j
    for i in range(1, compteur_cell + 1):
        J_matrix[i, 0] = J_10
        J_matrix[0, i] = J_10
        for j in range(1, compteur_cell + 1):
            if i != j:
                J_matrix[i, j] = J_11
    
    for _ in range(10000):
    
        i, j = np.random.randint(0, N, 2) #l'indice de la matrice à changer
        
        type_old = type_map[i,j]   #Au lieu de faire une copie de toute ma matrice avec np.copy, je fais juste une copie de l'élément intéresant
        num_old = num_map[i,j] #L'ancien numéro de la matrice
        
        di, dj = voisins_numero[np.random.randint(0,8)]
        ni, nj = (i + di) % N, (j + dj) % N #Choisir le nouveau voisin, aleatoireemten
        
        type_new, num_new = type_map[ni, nj], num_map[ni, nj]
        
        if num_new == num_old:
            continue        #Par soucis d'optimisation, on necalcule rien si on ne change rine
    
            
            
        V_c = np.array([np.sum(num_map == i) for i in range(1, compteur_cell + 1)])
        dV = np.zeros(compteur_cell)
    
        dV[int(num_old) - 1] -= 1
        dV[int(num_new) - 1] += 1
    
        dE_vol = lbda * np.sum((V_c + dV - Vol_cible_list)**2 - (V_c - Vol_cible_list)**2)
        
        # dE_int = 0
        # voisin_energie = num_map[i+1,j], num_map[i-1,j], num_map[i,j+1], num_map[i,j-1]
        # for pix in voisin_energie:
        #     if pix != type_old:
        #         dE_int += 2
        
        dE_int = 0
        for (dx, dy) in voisins_numero: 
            xi, xj = (i + dx) % N, (j + dy) % N #Les indices, avec CLP
            type_voisin, num_voisin = type_map[xi, xj], num_map[xi, xj]
            E_avant = J(type_old, num_old, type_voisin, num_voisin)     #RENOMMER PROPREMENT
            E_apres = J(type_new, num_new, type_voisin, num_voisin)
            
            dE_int += (E_apres - E_avant)
                
       
       
        dE = dE_vol + dE_int
    
        # --- Metropolis ---
        if dE <= 0 or np.random.rand() < np.exp(-dE / T_boltz):
            type_map[i,j] = type_new
            num_map[i,j] = num_new
    
            
            
    if i%100 == 0:
        
        plt.imshow(num_map)
        plt.show()

#%%







# @numba.njit
def increase_temps(type_map, num_map, nutr_map, Vol_cible_list, nb_change):

    
            
    '''1 Les nutriments diffusent '''
    nutr_map = diffusion_nutroment_solve(nutr_map, num_map, type_map)
    
    # compteur_cell = int(compteur_cell)
    '''2 Le volume cible change en fonction '''
    for n in range(1, 1 + 1):
        Vol_cible_list =  increase_vol_cible(nutr_map, num_map, type_map, n, Vol_cible_list)
        
    '''3 Eventuelles divisions cellulaires '''
    for n in range(1, 1 + 1):
        if dois_je_me_diviser(num_map, n) :
            num_map, Vol_cible_list, compteur_cell = division_cell_axis(num_map, n, Vol_cible_list, 1)  

    # for _ in range(nb_change):

                
    #     '''4 selection du pixel à changer, choix du nouveau type éventuel '''

    #     i, j = np.random.randint(0, N, 2) #l'indice de la matrice à changer
    
    #     type_old = int(M[i,j,0])   #Au lieu de faire une copie de toute ma matrice avec np.copy, je fais juste une copie de l'élément intéresant
    #     num_old = int(M[i,j,1]) #L'ancien numéro de la matric

    #     # choisir un voisin aléatoire
    #     di, dj = voisins_numero[np.random.randint(0,8)]
    #     ni, nj = (i + di) % N, (j + dj) % N #Choisir le nouveau voisin, aleatoireemten
        

    #     type_new = int(M[ni, nj, 0])
    #     num_new  = int(M[ni, nj, 1])

    #     if num_new == num_old:
    #         continue        #Par soucis d'optimisation, on necalcule rien si on ne change rine
            
    #     '''5 variation d'énergie '''

    #     dE_int = dE_surf_local(M, i, j, type_old, num_old, type_new, num_new)                
    #     dE_vol = dE_volume_local(V_c, V_cible, type_old, num_old, type_new, num_new)
        
    #     dE = dE_int + dE_vol
        
    #     '''6 Matropolis '''

    #     # --- Metropolis ---
    #     if dE <= 0 or np.random.random() < np.exp(-dE / T):
    #         # calculer les indices des cellules affectées AVANT de modifier V_c
    #         if num_old > 0:
    #             idx_old = (num_old - 1) if (type_old == 1) else (n1 + num_old - 1)
    #         else:
    #             idx_old = -1

    #         if num_new > 0:
    #             idx_new = (num_new - 1) if (type_new == 1) else (n1 + num_new - 1)
    #         else:
    #             idx_new = -1

    #         # appliquer le changement sur M
    #         M[i, j, 0] = type_new
    #         M[i, j, 1] = num_new

    #         # --- Mise à jour des volumes (in-place) ---
    #         if idx_old >= 0:
    #             V_c[idx_old] -= 1
    #         if idx_new >= 0:
    #             V_c[idx_new] += 1

    return type_map, num_map, nutr_map, Vol_cible_list

    
    
    
#%%

type_map = np.zeros((N, N), dtype=np.int32)
num_map = np.zeros((N, N), dtype=np.int32)
nutr_map = np.ones((N, N), dtype=np.float64) * S_outside
Vol_cible_list = [Vol_cible_init]  # vol cible de la première cellule

compteur_cell = 1
init()

def def_temp(type_map, nb_pas):
    type_map = np.copy(type_map)
    # summ = np.zeros(nb_pas)
    for i in range(nb_pas):
        increase_temps(type_map, num_map, nutr_map, Vol_cible_list, N*N)
        print(Vol_cible_list)
        # summ[i] = np.sum(M)
        if i % 1 == 0:
            # plot_type(M, time = i, cmap = 'inferno')
            plt.imshow(type_map, cmap = 'magma')
            plt.axis('off')
            # plt.colorbar()
            plt.show()
            # plot_num(M, time = i)
    # plt.plot(np.arange(0,nb_pas, 1), summ)
    # plt.xlabel('temps')
    # plt.ylabel('nombre de pixels 1')
        
def_temp(type_map, 100001)
    
    
    