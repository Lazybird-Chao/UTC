/*
 * task.h
 *
 *  Created on: Nov 14, 2017
 *      Author: chaoliu
 */

#ifndef TASK_H_
#define TASK_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Utc.h"
#include "common.h"

class SceneInit: public UserTaskBase{
public:
	void runImpl(char* filename,
			sphere_array_t *obj_array,
			global_vars *g_vars,
			vec3_t *lights ){
		FILE *fp = fopen((const char*)filename, "rb");
		if(fp ==NULL){
			std::cerr<<"Need a input scene file."<<std::endl;
			exit(1);
		}

		/*
		 * read scene file
		 */
		char line[256], *ptr, type;

		sphere_t* obj_list = (sphere_t*)malloc(sizeof(struct sphere));
		obj_list->next = NULL;
		int obj_count=0;
		camera_t cam;
		int lnum=0;
#define DELIM	" \t\n"
		while((ptr = fgets(line, 256, fp))) {
			int i;
			vec3_t pos, col;
			FTYPE rad, spow, refl;

			while(*ptr == ' ' || *ptr == '\t') ptr++;
			if(*ptr == '#' || *ptr == '\n') continue;

			if(!(ptr = strtok(line, DELIM))) continue;
			type = *ptr;

			for(i=0; i<3; i++) {
				if(!(ptr = strtok(0, DELIM))) break;
				*((FTYPE*)&pos.x + i) = (FTYPE)atof(ptr);
			}

			if(type == 'l') {
				lights[lnum++] = pos;
				continue;
			}

			if(!(ptr = strtok(0, DELIM))) continue;
			rad = atof(ptr);

			for(i=0; i<3; i++) {
				if(!(ptr = strtok(0, DELIM))) break;
				*((FTYPE*)&col.x + i) = (FTYPE)atof(ptr);
			}

			if(type == 'c') {
				cam.pos = pos;
				cam.targ = col;
				cam.fov = rad;
				continue;
			}

			if(!(ptr = strtok(0, DELIM))) continue;
			spow = (FTYPE)atof(ptr);

			if(!(ptr = strtok(0, DELIM))) continue;
			refl = (FTYPE)atof(ptr);

			if(type == 's') {
				obj_count++;
				struct sphere *sph = (sphere_t*)malloc(sizeof *sph);
				sph->next = obj_list->next;
				obj_list->next = sph;

				sph->pos = pos;
				sph->rad = rad;
				sph->mat.col = col;
				sph->mat.spow = spow;
				sph->mat.refl = refl;
			} else {
				fprintf(stderr, "unknown type: %c\n", type);
			}
		}
		fclose(fp);
		/*
		 * change the sphere linked list to an array
		 */
		obj_array->pos = (vec3_t*)malloc(sizeof(vec3_t)*obj_count);
		obj_array->mat = (material_t*)malloc(sizeof(material_t)*obj_count);
		obj_array->rad = (FTYPE*)malloc(sizeof(FTYPE)*obj_count);
		sphere_t *p1 = obj_list->next;
		sphere_t *p2 = p1;
		int i=0;
		while(p1!=NULL){
			obj_array->pos[i] = p1->pos;
			obj_array->rad[i] = p1->rad;
			obj_array->mat[i].col = p1->mat.col;
			obj_array->mat[i].spow = p1->mat.spow;
			obj_array->mat[i].refl = p1->mat.refl;
			p2 = p1;
			p1 = p1->next;
			free(p2);
			i++;
		}
		obj_list->next = NULL;
		free(obj_list);

		g_vars->cam = cam;
		g_vars->lnum = lnum;
		g_vars->obj_count = obj_count;

		if(__numGroupProcesses > 1){
			TaskBcastBy<char, 0>(this, g_vars, sizeof(*g_vars), 0);
			TaskBcastBy<char, 0>(this, obj_array->pos, sizeof(vec3_t)*obj_count, 0);
			TaskBcastBy<char, 0>(this, obj_array->mat, sizeof(material_t)*obj_count, 0);
			TaskBcastBy<char, 0>(this, obj_array->rad, sizeof(FTYPE)*obj_count, 0);
		}
		inter_Barrier();
	}
};





#endif /* TASK_H_ */
