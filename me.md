Neither SDF nor MultiFabGrid store _any_ grids or arrays in whatever format.<br>
SDF is lazy, and MultiFabGrid only provides a context in which multifab operations are performed or new multifabs can be created.

Todo:
- make SDF.to_multifab() accept a MultiFabGrid as an argument instead of the stuff we have now
- maybe remove the SDF.to_multifab() method?
- wtf is the point of _Array; just call it ndarray?