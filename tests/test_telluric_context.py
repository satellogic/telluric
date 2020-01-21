from telluric.context import local_context, TelluricContext


def test_context_in_one_level():
    with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
        assert local_context.get('a') == 1
        assert local_context.get('b') == 2
        assert local_context.get('c') == 'stam'
        assert local_context.get('d') == {'a': 'a', 'b': 'b'}
    assert local_context._options == {}


def test_context_in_two_level():
    with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
        with TelluricContext(a=4, b=5, x='something', y=13):
            assert local_context.get('a') == 4
            assert local_context.get('b') == 5
            assert local_context.get('c') == 'stam'
            assert local_context.get('d') == {'a': 'a', 'b': 'b'}
            assert local_context.get('x') == 'something'
            assert local_context.get('y') == 13
        assert local_context.get('a') == 1
        assert local_context.get('b') == 2
        assert local_context.get('c') == 'stam'
        assert local_context.get('d') == {'a': 'a', 'b': 'b'}
    assert local_context._options == {}


def test_different_context_on_different_threads():
    import threading
    from time import sleep

    def thread_test_1():
        with TelluricContext(a=1, b=2, c='stam', d={'a': 'a', 'b': 'b'}):
            sleep(0.1)
            assert local_context.get('a') == 1
            assert local_context.get('b') == 2
            assert local_context.get('c') == 'stam'
            assert local_context.get('d') == {'a': 'a', 'b': 'b'}
            assert local_context.get('x') is None
            assert local_context.get('y') is None
        assert local_context._options == {}

    def thread_test_2():
        with TelluricContext(a=4, b=5, x='something', y=13):
            assert local_context.get('a') == 4
            assert local_context.get('b') == 5
            assert local_context.get('x') == 'something'
            assert local_context.get('y') == 13
            assert local_context.get('c') is None
            assert local_context.get('d') is None
            sleep(0.1)
        assert local_context._options == {}

    t1 = threading.Thread(target=thread_test_1)
    t2 = threading.Thread(target=thread_test_2)
    t1.start()
    t2.start()
    assert local_context._options == {}
    t1.join()
    t2.join()
    assert local_context._options == {}
