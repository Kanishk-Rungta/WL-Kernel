METHODS={'original':('uniform','original'),'adaptive':('degree_adaptive','original'),'augmented':('uniform','degree_augmented'),'combined':('degree_adaptive','degree_augmented')}
METHODS.update(self_calibrated=('self_calibrated','original'),
               self_calibrated_augmented=('self_calibrated','degree_augmented'))
PHASE1_METHODS = ['original', 'adaptive', 'augmented', 'combined']
PHASE2_METHODS = ['original', 'adaptive', 'augmented', 'self_calibrated', 'self_calibrated_augmented']
