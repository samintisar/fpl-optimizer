// C++ operator new/delete through mimalloc (as the Rust side's global allocator is): HiGHS
// allocates on every backtest thread at once, and the Windows heap serializes them.
// Linked only because src/lib.rs calls fplopt_highs_new_delete().
#include <mimalloc-new-delete.h>

extern "C" void fplopt_highs_new_delete(void) {}
