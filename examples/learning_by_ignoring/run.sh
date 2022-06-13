# Cl -> Ar
python main.py --gpu=0 --source_domain=Cl --target_domain=Ar --lam=7e-3 --baseline
python main.py --gpu=0 --source_domain=Cl --target_domain=Ar --lam=7e-3

# Ar -> Pr
python main.py --gpu=0 --source_domain=Ar --target_domain=Pr --lam=7e-3 --baseline
python main.py --gpu=0 --source_domain=Ar --target_domain=Pr --lam=7e-3

# Pr -> Rw
python main.py --gpu=0 --source_domain=Pr --target_domain=Rw --lam=7e-3 --baseline
python main.py --gpu=0 --source_domain=Pr --target_domain=Rw --lam=7e-3

# Rw -> Cl
python main.py --gpu=0 --source_domain=Rw --target_domain=Cl --lam=7e-3 --baseline
python main.py --gpu=0 --source_domain=Rw --target_domain=Cl --lam=7e-3
