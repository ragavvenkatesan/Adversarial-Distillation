            """ More layers in judge """ 
            par = None
            if input_params is not None:
                par = input_params
            
            fc1_out, params = dot_product_layer  (  input = self.images,
                                                    neurons = JUDGE,
                                                    params = par,
                                                    name = 'dot')
            process_params(params, name = self.name)
            
            # Embedding layers
            expert_embed, params = dot_product_layer (input = expert,   
                                              neurons = EMBED,
                                              name = 'expert_embed' )
            novice_embed, params = dot_product_layer (input = novice,
                                              neurons = EMBED,
                                              params = params,
                                              name = 'novice_embed' )
            process_params ( params, name = self.name )
            


